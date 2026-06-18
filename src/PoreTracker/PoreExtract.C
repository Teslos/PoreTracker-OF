/*---------------------------------------------------------------------------*\
    PoreTracker-OF  —  Pore-scale tracking functionObject for OpenFOAM
\*---------------------------------------------------------------------------*/

#include "PoreExtract.H"
#include "addToRunTimeSelectionTable.H"
#include "processorPolyPatch.H"
#include "PstreamBuffers.H"
#include "UOPstream.H"
#include "UIPstream.H"
#include "Time.H"
#include "fvMesh.H"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(PoreExtract, 0);
    addToRunTimeSelectionTable(functionObject, PoreExtract, dictionary);
}
}

// * * * * * * * * * * * * * * Local DSU helper * * * * * * * * * * * * * * //

namespace
{

struct DSU
{
    std::vector<int> parent, rank_;

    explicit DSU(int n) : parent(n), rank_(n, 0)
    { std::iota(parent.begin(), parent.end(), 0); }

    int find(int x)
    {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }

    void unite(int a, int b)
    {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }
};

// Serialise a pore map to a flat scalar list.
// Layout: [ nPores, (gID vol cx cy cz fx fy fz nc) x nPores ]
// Note: centroid is stored raw (volume-weighted sum) during the reduction.
Foam::List<Foam::scalar> packPores
(
    const std::map<Foam::label, Foam::functionObjects::PoreData>& pores
)
{
    const Foam::label n = pores.size();
    Foam::List<Foam::scalar> out(1 + n * 9);
    Foam::label i = 0;
    out[i++] = Foam::scalar(n);
    for (const auto& kv : pores)
    {
        const auto& p = kv.second;
        out[i++] = Foam::scalar(p.globalID);
        out[i++] = p.volume;
        out[i++] = p.centroid.x();
        out[i++] = p.centroid.y();
        out[i++] = p.centroid.z();
        out[i++] = p.force.x();
        out[i++] = p.force.y();
        out[i++] = p.force.z();
        out[i++] = Foam::scalar(p.nCells);
    }
    return out;
}

// Accumulate a packed list into an existing map (additive — for partial sums).
void accumPacked
(
    const Foam::List<Foam::scalar>& packed,
    std::map<Foam::label, Foam::functionObjects::PoreData>& pores
)
{
    if (packed.empty()) return;
    Foam::label n = Foam::label(packed[0]);
    Foam::label i = 1;
    for (Foam::label k = 0; k < n; ++k)
    {
        Foam::label gID = Foam::label(packed[i]);
        auto& p = pores[gID];
        p.globalID  = gID;
        p.volume   += packed[i+1];
        p.centroid += Foam::vector(packed[i+2], packed[i+3], packed[i+4]);
        p.force    += Foam::vector(packed[i+5], packed[i+6], packed[i+7]);
        p.nCells   += Foam::label(packed[i+8]);
        i += 9;
    }
}

// Unpack a packed list (replacement, not additive — for already-finalised data).
void unpackPores
(
    const Foam::List<Foam::scalar>& packed,
    std::map<Foam::label, Foam::functionObjects::PoreData>& pores
)
{
    pores.clear();
    if (packed.empty()) return;
    Foam::label n = Foam::label(packed[0]);
    Foam::label i = 1;
    for (Foam::label k = 0; k < n; ++k)
    {
        Foam::label gID = Foam::label(packed[i]);
        auto& p = pores[gID];
        p.globalID  = gID;
        p.volume    = packed[i+1];
        p.centroid  = Foam::vector(packed[i+2], packed[i+3], packed[i+4]);
        p.force     = Foam::vector(packed[i+5], packed[i+6], packed[i+7]);
        p.nCells    = Foam::label(packed[i+8]);
        i += 9;
    }
}

} // anonymous namespace


// * * * * * * * * * * * * * * * * Constructor * * * * * * * * * * * * * * * //

Foam::functionObjects::PoreExtract::PoreExtract
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    alphaName_("alpha.gas"),
    TName_("T"),
    fLorentzName_("fLorentz"),
    alphaThreshold_(0.5),
    alphaIsGas_(true),
    solidusTemp_(1700),
    searchRadius_(5e-4),
    overlapThreshold_(0.3),
    exclusionScale_(0.75),
    vLaser_(Zero),
    laserOrigin_(Zero),
    runPythonAtEnd_(false),
    pythonScript_("scripts/PoreStats.py"),
    nextPoreID_(1),
    executionCount_(0),
    prevTrackIDPerCell_(mesh_.nCells(), -1)
{
    read(dict);
    openCSV();
}


Foam::functionObjects::PoreExtract::~PoreExtract()
{
    if (runPythonAtEnd_ && Pstream::master())
    {
        const std::string csvPath =
            (mesh_.time().path() / (name() + "_pores.csv")).c_str();
        const std::string cmd =
            "python3 " + std::string(pythonScript_.c_str())
            + " " + csvPath
            + " --vLaser "
            + Foam::name(vLaser_.x()) + " "
            + Foam::name(vLaser_.y()) + " "
            + Foam::name(vLaser_.z());
        Info << "PoreTracker: running post-processing: " << cmd.c_str() << nl;
        const int rc = ::system(cmd.c_str());
        if (rc != 0)
        {
            WarningInFunction
                << "Post-processing command failed with return code " << rc
                << nl;
        }
    }
}


// * * * * * * * * * * * * * * * * read() * * * * * * * * * * * * * * * * * //

bool Foam::functionObjects::PoreExtract::read(const dictionary& dict)
{
    fvMeshFunctionObject::read(dict);

    dict.readIfPresent("alphaName",        alphaName_);
    dict.readIfPresent("TName",            TName_);
    dict.readIfPresent("fLorentzName",     fLorentzName_);
    dict.readIfPresent("alphaThreshold",   alphaThreshold_);
    dict.readIfPresent("alphaIsGas",       alphaIsGas_);
    dict.readIfPresent("solidusTemp",      solidusTemp_);
    dict.readIfPresent("searchRadius",     searchRadius_);
    dict.readIfPresent("overlapThreshold", overlapThreshold_);
    dict.readIfPresent("exclusionScale",   exclusionScale_);
    dict.readIfPresent("vLaser",           vLaser_);
    dict.readIfPresent("laserOrigin",      laserOrigin_);
    dict.readIfPresent("runPythonAtEnd",   runPythonAtEnd_);
    dict.readIfPresent("pythonScript",     pythonScript_);

    return true;
}


// * * * * * * * * * * * * * Phase 1 : local CCL * * * * * * * * * * * * * * //

Foam::List<Foam::label>
Foam::functionObjects::PoreExtract::localCCL
(
    const scalarField& alpha,
    const scalarField& T
) const
{
    const label nCells = mesh_.nCells();
    List<label> compID(nCells, -1);

    // Active cell: gas pore inside the liquid melt pool
    boolList active(nCells, false);
    forAll(alpha, cellI)
    {
        const bool cellIsGas = alphaIsGas_
            ? (alpha[cellI] > alphaThreshold_)
            : (alpha[cellI] < (1.0 - alphaThreshold_));
        if (cellIsGas && T[cellI] > solidusTemp_)
            active[cellI] = true;
    }

    const labelListList& nbrs = mesh_.cellCells();
    label nComp = 0;
    DynamicList<label> queue;

    forAll(active, seed)
    {
        if (!active[seed] || compID[seed] != -1) continue;

        queue.clear();
        queue.append(seed);
        compID[seed] = nComp;

        for (label qi = 0; qi < queue.size(); ++qi)
        {
            for (const label nbr : nbrs[queue[qi]])
            {
                if (active[nbr] && compID[nbr] == -1)
                {
                    compID[nbr] = nComp;
                    queue.append(nbr);
                }
            }
        }
        ++nComp;
    }

    return compID;
}


// * * * * * * * * * * Phase 2 : parallel boundary resolution * * * * * * * //
//
// Algorithm:
//   1. Each proc computes local component count.
//   2. Gather counts → compute sequential proc offsets.
//   3. Exchange boundary-cell component IDs with neighbours.
//   4. Gather merge pairs (seqID_A, seqID_B) to master.
//   5. Master runs DSU, assigns monotonically increasing global pore IDs.
//   6. Master scatters per-proc (localComp → globalID) mapping.

void Foam::functionObjects::PoreExtract::resolveParallelBoundaries
(
    const List<label>& localCompID,
    List<label>&        globalCompID
) const
{
    const label nCells = mesh_.nCells();
    globalCompID.setSize(nCells, -1);

    // --- count local components ---
    label nLocalComp = 0;
    forAll(localCompID, i)
        if (localCompID[i] >= 0)
            nLocalComp = max(nLocalComp, localCompID[i] + 1);

    if (!Pstream::parRun())
    {
        globalCompID = localCompID;
        return;
    }

    const label myProc = Pstream::myProcNo();
    const label nProcs = Pstream::nProcs();

    // --- gather all local component counts ---
    labelList allNLocal(nProcs, 0);
    allNLocal[myProc] = nLocalComp;
    Pstream::gatherList(allNLocal);
    Pstream::scatterList(allNLocal);

    // sequential offset for this proc's component IDs in the master DSU
    label myOffset = 0;
    for (label p = 0; p < myProc; ++p) myOffset += allNLocal[p];

    label totalNodes = 0;
    forAll(allNLocal, p) totalNodes += allNLocal[p];

    // proc offset array (needed on master later)
    labelList procOffset(nProcs, 0);
    for (label p = 1; p < nProcs; ++p)
        procOffset[p] = procOffset[p-1] + allNLocal[p-1];

    // --- exchange boundary component IDs with neighbours ---
    // Use non-blocking PstreamBuffers: post all sends before any receives to
    // avoid the send/receive ordering deadlock that arises with blocking streams.
    List<labelPair> localMergePairs;

    {
        PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);

        // Pass 1: pack and send to every processor-patch neighbour
        forAll(mesh_.boundaryMesh(), patchI)
        {
            const polyPatch& pp = mesh_.boundaryMesh()[patchI];
            if (!isA<processorPolyPatch>(pp)) continue;

            const processorPolyPatch& procPatch =
                refCast<const processorPolyPatch>(pp);
            const label nbrProc = procPatch.neighbProcNo();
            const labelList& faceCells = procPatch.faceCells();

            labelList myIDs(faceCells.size());
            forAll(faceCells, fi)
                myIDs[fi] = localCompID[faceCells[fi]];  // may be -1

            UOPstream toNbr(nbrProc, pBufs);
            toNbr << myIDs;
        }

        pBufs.finishedSends();   // flush all sends; safe to receive now

        // Pass 2: receive and build merge pairs
        forAll(mesh_.boundaryMesh(), patchI)
        {
            const polyPatch& pp = mesh_.boundaryMesh()[patchI];
            if (!isA<processorPolyPatch>(pp)) continue;

            const processorPolyPatch& procPatch =
                refCast<const processorPolyPatch>(pp);
            const label nbrProc = procPatch.neighbProcNo();
            const label nbrOffset = procOffset[nbrProc];
            const labelList& faceCells = procPatch.faceCells();

            labelList myIDs(faceCells.size());
            forAll(faceCells, fi)
                myIDs[fi] = localCompID[faceCells[fi]];

            labelList nbrIDs;
            UIPstream fromNbr(nbrProc, pBufs);
            fromNbr >> nbrIDs;

            forAll(myIDs, fi)
            {
                if (myIDs[fi] >= 0 && nbrIDs[fi] >= 0)
                {
                    label seqA = myOffset  + myIDs[fi];
                    label seqB = nbrOffset + nbrIDs[fi];
                    if (seqA != seqB)
                        localMergePairs.append(labelPair(seqA, seqB));
                }
            }
        }
    }

    // --- gather all merge pairs to master ---
    List<List<labelPair>> allMergePairs(nProcs);
    allMergePairs[myProc] = localMergePairs;
    Pstream::gatherList(allMergePairs);

    // per-proc mapping: localComp → globalPoreID
    List<label> myGlobalMap(nLocalComp, -1);

    if (Pstream::master())
    {
        // DSU over all sequential IDs
        DSU dsu(max(1, totalNodes));

        for (const auto& pairList : allMergePairs)
            for (const auto& mp : pairList)
                dsu.unite(int(mp.first()), int(mp.second()));

        // Assign monotonically increasing global pore IDs (1-based; 0 = keyhole later)
        std::map<int, label> rootToGlobal;
        label nextGlobal = 1;

        // Build per-proc maps and buffer for scatter
        List<List<label>> allGlobalMaps(nProcs);
        for (label p = 0; p < nProcs; ++p)
        {
            allGlobalMaps[p].setSize(allNLocal[p], -1);
            for (label lc = 0; lc < allNLocal[p]; ++lc)
            {
                int root = dsu.find(int(procOffset[p] + lc));
                auto it = rootToGlobal.find(root);
                if (it == rootToGlobal.end())
                {
                    rootToGlobal[root] = nextGlobal++;
                    it = rootToGlobal.find(root);
                }
                allGlobalMaps[p][lc] = it->second;
            }
        }

        myGlobalMap = allGlobalMaps[0];

        // Send per-proc maps to non-masters
        for (label p = 1; p < nProcs; ++p)
        {
            OPstream toP(Pstream::commsTypes::blocking, p);
            toP << allGlobalMaps[p];
        }
    }
    else
    {
        IPstream fromMaster(Pstream::commsTypes::blocking, Pstream::masterNo());
        fromMaster >> myGlobalMap;
    }

    // --- apply mapping to all cells ---
    forAll(localCompID, cellI)
    {
        const label lc = localCompID[cellI];
        if (lc >= 0 && lc < myGlobalMap.size())
            globalCompID[cellI] = myGlobalMap[lc];
    }
}


// * * * * * * * * * * Phase 3 : pore property computation * * * * * * * * * //

std::map<Foam::label, Foam::functionObjects::PoreData>
Foam::functionObjects::PoreExtract::computePoreProperties
(
    const List<label>& globalCompID,
    const vectorField& fLorentz
) const
{
    std::map<label, PoreData> pores;

    const scalarField& vols  = mesh_.V();
    const vectorField& ctrs  = mesh_.C();

    // Local accumulation
    forAll(globalCompID, cellI)
    {
        const label gID = globalCompID[cellI];
        if (gID < 0) continue;

        PoreData& p = pores[gID];
        p.globalID  = gID;
        const scalar v = vols[cellI];
        p.volume   += v;
        p.centroid += v * ctrs[cellI];       // raw volume-weighted sum
        p.force    -= exclusionScale_ * v * fLorentz[cellI];  // exclusion opposes Lorentz
        p.nCells   += 1;
    }

    if (Pstream::parRun())
    {
        // Gather partial sums to master
        List<List<scalar>> allPacked(Pstream::nProcs());
        allPacked[Pstream::myProcNo()] = packPores(pores);
        Pstream::gatherList(allPacked);

        if (Pstream::master())
        {
            pores.clear();
            for (const auto& packed : allPacked)
                accumPacked(packed, pores);

            // Finalise centroids: divide weighted sum by total volume
            for (auto& kv : pores)
                if (kv.second.volume > VSMALL)
                    kv.second.centroid /= kv.second.volume;

            // Mark keyhole: largest-volume pore at z_min (confirmed by largest V)
            label keyholeGID = -1;
            scalar maxVol = -GREAT;
            for (const auto& kv : pores)
                if (kv.second.volume > maxVol)
                { maxVol = kv.second.volume; keyholeGID = kv.first; }
            if (keyholeGID >= 0)
                pores[keyholeGID].isKeyhole = true;

            // Broadcast finalised map to all non-masters
            List<scalar> masterPacked = packPores(pores);
            // also send isKeyhole flag — append it as (gID, 1) pairs
            label nK = 0;
            for (const auto& kv : pores) if (kv.second.isKeyhole) ++nK;
            masterPacked.append(scalar(nK));
            for (const auto& kv : pores)
                if (kv.second.isKeyhole)
                    masterPacked.append(scalar(kv.first));

            for (label p = 1; p < Pstream::nProcs(); ++p)
            {
                OPstream toP(Pstream::commsTypes::blocking, p);
                toP << masterPacked;
            }
        }
        else
        {
            List<scalar> masterPacked;
            IPstream fromMaster(Pstream::commsTypes::blocking, Pstream::masterNo());
            fromMaster >> masterPacked;

            // Strip keyhole flags appended at the tail
            // Find where the pore block ends: first 1 + n*9 elements
            label n = label(masterPacked[0]);
            label baseSize = 1 + n * 9;
            List<scalar> poreBlock(baseSize);
            for (label i = 0; i < baseSize; ++i)
            {
                poreBlock[i] = masterPacked[i];
            }
            unpackPores(poreBlock, pores);

            // Finalise centroids (already done by master, but centroid was packed finalised)
            // Apply keyhole flags
            if (masterPacked.size() > baseSize)
            {
                label nK = label(masterPacked[baseSize]);
                for (label k = 0; k < nK; ++k)
                {
                    label kID = label(masterPacked[baseSize + 1 + k]);
                    if (pores.count(kID)) pores[kID].isKeyhole = true;
                }
            }
        }
    }
    else
    {
        // Serial path: finalise in-place
        for (auto& kv : pores)
            if (kv.second.volume > VSMALL)
                kv.second.centroid /= kv.second.volume;

        label keyholeGID = -1;
        scalar maxVol = -GREAT;
        for (const auto& kv : pores)
            if (kv.second.volume > maxVol)
            { maxVol = kv.second.volume; keyholeGID = kv.first; }
        if (keyholeGID >= 0)
            pores[keyholeGID].isKeyhole = true;
    }

    return pores;
}


// * * * * * * * * * * * Phase 4 : tracking & ID matching * * * * * * * * * //

Foam::label Foam::functionObjects::PoreExtract::findNearestActivePore
(
    const vector& pos,
    const std::set<label>& excluded
) const
{
    label bestID   = -1;
    scalar bestD2  = searchRadius_ * searchRadius_;

    for (const auto& kv : activePores_)
    {
        if (excluded.count(kv.first)) continue;
        scalar d2 = magSqr(kv.second.centroid - pos);
        if (d2 < bestD2) { bestD2 = d2; bestID = kv.first; }
    }
    return bestID;
}


void Foam::functionObjects::PoreExtract::matchAndUpdatePores
(
    std::map<label, PoreData>& newPores,
    const std::map<label, std::map<label, label>>& overlapCounts
)
{
    const scalar t = mesh_.time().value();

    if (activePores_.empty())
    {
        // First call — assign initial tracking IDs
        for (auto& kv : newPores)
        {
            PoreData& p = kv.second;
            label trackID = p.isKeyhole ? 0 : nextPoreID_++;
            p.globalID      = trackID;
            p.birthTime     = t;
            p.birthCentroid = p.centroid;
        }
        // Rebuild map keyed by trackID
        std::map<label, PoreData> tmp;
        for (auto& kv : newPores) tmp[kv.second.globalID] = kv.second;
        activePores_ = std::move(tmp);
        return;
    }

    // Sort new pores by volume desc so largest pores claim IDs first
    std::vector<label> newKeys;
    newKeys.reserve(newPores.size());
    for (const auto& kv : newPores) newKeys.push_back(kv.first);
    std::sort(newKeys.begin(), newKeys.end(),
        [&](label a, label b){ return newPores[a].volume > newPores[b].volume; });

    std::set<label> usedOldIDs;

    for (label nk : newKeys)
    {
        PoreData& np = newPores[nk];

        if (np.isKeyhole)
        {
            // Keyhole always gets ID 0
            usedOldIDs.insert(0);
            const auto it = activePores_.find(0);
            if (it != activePores_.end())
            {
                np.birthTime     = it->second.birthTime;
                np.birthCentroid = it->second.birthCentroid;
            }
            else
            {
                np.birthTime     = t;
                np.birthCentroid = np.centroid;
            }
            np.globalID = 0;
            continue;
        }

        label matchedID = -1;

        // 1) Primary matching: overlap with previously tracked cells.
        //    Keep an ID only if overlap ratio passes overlapThreshold_.
        auto ocIt = overlapCounts.find(nk);
        if (ocIt != overlapCounts.end() && np.nCells > 0)
        {
            label bestOld = -1;
            label bestCount = -1;
            scalar bestOldVol = -GREAT;

            for (const auto& oldPair : ocIt->second)
            {
                const label candidate = oldPair.first;
                if (usedOldIDs.count(candidate)) continue;

                const label c = oldPair.second;
                if (c > bestCount)
                {
                    bestCount = c;
                    bestOld = candidate;
                    const auto oldIt = activePores_.find(candidate);
                    bestOldVol = (oldIt != activePores_.end()) ? oldIt->second.volume : -GREAT;
                }
                else if (c == bestCount)
                {
                    const auto oldIt = activePores_.find(candidate);
                    const scalar candidateVol =
                        (oldIt != activePores_.end()) ? oldIt->second.volume : -GREAT;
                    if (candidateVol > bestOldVol)
                    {
                        bestOld = candidate;
                        bestOldVol = candidateVol;
                    }
                }
            }

            if (bestOld >= 0)
            {
                const scalar overlapRatio = scalar(bestCount)/scalar(np.nCells);
                if (overlapRatio >= overlapThreshold_)
                {
                    matchedID = bestOld;
                }
            }
        }

        // 2) Fallback: nearest previously active pore within searchRadius_.
        if (matchedID == -1)
        {
            matchedID = findNearestActivePore(np.centroid, usedOldIDs);
        }

        if (matchedID != -1)
        {
            usedOldIDs.insert(matchedID);
            np.globalID      = matchedID;
            np.birthTime     = activePores_[matchedID].birthTime;
            np.birthCentroid = activePores_[matchedID].birthCentroid;
        }
        else
        {
            // New pore: birth event or keyhole pinch-off
            label newID = nextPoreID_++;
            usedOldIDs.insert(newID);
            np.globalID      = newID;
            np.birthTime     = t;
            np.birthCentroid = np.centroid;
            if (Pstream::master())
                Info << "PoreTracker: pore birth  ID=" << newID
                     << "  t=" << t
                     << "  centroid=" << np.centroid << nl;
        }
    }

    // Report pores that vanished (solidified)
    for (const auto& kv : activePores_)
    {
        if (!usedOldIDs.count(kv.first) && Pstream::master())
        {
            Info << "PoreTracker: pore death  ID=" << kv.first
                 << "  t=" << t << nl;
        }
    }

    // Rebuild activePores_ keyed by persistent trackID
    activePores_.clear();
    for (auto& kv : newPores)
    {
        activePores_[kv.second.globalID] = kv.second;
    }
}


// * * * * * * * * * * * * * * CSV output * * * * * * * * * * * * * * * * * //

void Foam::functionObjects::PoreExtract::openCSV()
{
    if (!Pstream::master()) return;

    const fileName csvPath =
        mesh_.time().path() / (name() + "_pores.csv");

    csvPtr_.reset(new std::ofstream(csvPath.c_str()));
    *csvPtr_
        << "Time,PoreID,IsKeyhole,Volume_m3,"
        << "Cx_m,Cy_m,Cz_m,"
        << "Fx_N,Fy_N,Fz_N,"
        << "BirthTime,BirthCx,BirthCy,BirthCz,"
        << "NCells\n";
    csvPtr_->flush();
}


void Foam::functionObjects::PoreExtract::writeCSVRow
(
    scalar t,
    const PoreData& p
) const
{
    if (!Pstream::master() || !csvPtr_) return;

    *csvPtr_ << std::setprecision(10)
        << t                   << ","
        << p.globalID          << ","
        << (p.isKeyhole ? 1:0) << ","
        << p.volume            << ","
        << p.centroid.x()      << ","
        << p.centroid.y()      << ","
        << p.centroid.z()      << ","
        << p.force.x()         << ","
        << p.force.y()         << ","
        << p.force.z()         << ","
        << p.birthTime         << ","
        << p.birthCentroid.x() << ","
        << p.birthCentroid.y() << ","
        << p.birthCentroid.z() << ","
        << p.nCells            << "\n";
}


// * * * * * * * * * * * * * * * execute() * * * * * * * * * * * * * * * * * //
// Called every time step. Runs the full pipeline:
// CCL → parallel stitch → property computation → tracking.
// Write to CSV is deferred to write() which fires at writeInterval.

bool Foam::functionObjects::PoreExtract::execute()
{
    ++executionCount_;

    // --- get required fields ---
    // Support both live-run (fields already in objectRegistry) and postProcess
    // mode (registry is empty — read fields directly from the time directory).
    auto fetchScalar =
        [&](const word& name, tmp<volScalarField>& store) -> const volScalarField&
    {
        if (mesh_.foundObject<volScalarField>(name))
            return mesh_.lookupObject<volScalarField>(name);
        store = tmp<volScalarField>::New(
            IOobject(name, mesh_.time().timeName(), mesh_,
                     IOobject::MUST_READ, IOobject::NO_WRITE),
            mesh_);
        return store();
    };

    tmp<volScalarField> tAlpha, tT;
    const volScalarField& alpha = fetchScalar(alphaName_, tAlpha);
    const volScalarField& T     = fetchScalar(TName_,     tT);

    // fLorentz is optional (zero if absent); try disk read in postProcess mode
    tmp<volVectorField> tFLorentz;
    vectorField zeroVF(mesh_.nCells(), Zero);
    const vectorField* fLorenzPtr = &zeroVF;
    if (mesh_.foundObject<volVectorField>(fLorentzName_))
    {
        fLorenzPtr = &mesh_.lookupObject<volVectorField>(fLorentzName_).primitiveField();
    }
    else
    {
        IOobject ioF(fLorentzName_, mesh_.time().timeName(), mesh_,
                     IOobject::READ_IF_PRESENT, IOobject::NO_WRITE);
        if (ioF.typeHeaderOk<volVectorField>(false))
        {
            tFLorentz = tmp<volVectorField>::New(ioF, mesh_);
            fLorenzPtr = &tFLorentz().primitiveField();
        }
    }

    // --- Phase 1: local CCL ---
    const List<label> localComp =
        localCCL(alpha.primitiveField(), T.primitiveField());

    // --- Phase 2: parallel stitch → globally unique comp IDs ---
    List<label> globalComp;
    resolveParallelBoundaries(localComp, globalComp);

    // --- Phase 3: pore properties (volume, centroid, force) ---
    std::map<label, PoreData> newPores =
        computePoreProperties(globalComp, *fLorenzPtr);

    // Build overlap map input by translating new connected-component IDs into
    // current step internal keys (newPores map keys are the same IDs here).
    std::map<label, std::map<label, label>> overlapCounts;
    forAll(globalComp, cellI)
    {
        const label newKey = globalComp[cellI];
        if (newKey < 0) continue;
        if (cellI >= prevTrackIDPerCell_.size()) continue;
        const label oldTrack = prevTrackIDPerCell_[cellI];
        if (oldTrack < 0) continue;
        overlapCounts[newKey][oldTrack] += 1;
    }

    // --- Phase 4: match IDs to previous time step ---
    matchAndUpdatePores(newPores, overlapCounts);

    // Persist per-cell track IDs for overlap matching at next execute().
    if (prevTrackIDPerCell_.size() != mesh_.nCells())
    {
        prevTrackIDPerCell_.setSize(mesh_.nCells(), -1);
    }
    forAll(prevTrackIDPerCell_, cellI)
    {
        prevTrackIDPerCell_[cellI] = -1;
        const label comp = globalComp[cellI];
        if (comp < 0) continue;
        const auto it = newPores.find(comp);
        if (it != newPores.end())
        {
            prevTrackIDPerCell_[cellI] = it->second.globalID;
        }
    }

    return true;
}


// * * * * * * * * * * * * * * * * write() * * * * * * * * * * * * * * * * * //
// Called at writeInterval. Appends current pore states to CSV.

bool Foam::functionObjects::PoreExtract::write()
{
    const scalar t = mesh_.time().value();

    if (Pstream::master())
        Info << "PoreTracker: write at t=" << t
             << "  nPores=" << activePores_.size() << nl;

    for (const auto& kv : activePores_)
        writeCSVRow(t, kv.second);

    if (Pstream::master() && csvPtr_)
        csvPtr_->flush();

    return true;
}


bool Foam::functionObjects::PoreExtract::end()
{
    write();
    return true;
}
