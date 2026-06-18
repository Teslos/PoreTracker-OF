/*---------------------------------------------------------------------------*\
    PoreTracker-OF  —  Pore-scale tracking functionObject for OpenFOAM
    Tracks gas pores inside a solidifying melt pool, assigns persistent IDs,
    computes Lorentz-induced exclusion forces, and streams results to CSV.
\*---------------------------------------------------------------------------*/

#ifndef PoreExtract_H
#define PoreExtract_H

#include "fvMeshFunctionObject.H"
#include "volFields.H"
#include "OFstream.H"
#include "autoPtr.H"

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <fstream>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                          Struct PoreData
\*---------------------------------------------------------------------------*/

struct PoreData
{
    label   globalID;
    scalar  volume;
    vector  centroid;       // volume-weighted (raw) during accumulation; finalised after
    vector  force;          // exclusion force = -exclusionScale * Sum(V_cell * fLorentz)
    vector  birthCentroid;
    scalar  birthTime;
    bool    isKeyhole;
    label   nCells;

    PoreData()
    :
        globalID(-1),
        volume(0),
        centroid(Zero),
        force(Zero),
        birthCentroid(Zero),
        birthTime(0),
        isKeyhole(false),
        nCells(0)
    {}
};


/*---------------------------------------------------------------------------*\
                        Class PoreExtract Declaration
\*---------------------------------------------------------------------------*/

class PoreExtract
:
    public fvMeshFunctionObject
{
    // ---- configuration ----
    word    alphaName_;
    word    TName_;
    word    fLorentzName_;
    scalar  alphaThreshold_;
    bool    alphaIsGas_;    // true: field is gas-phase fraction (high = gas); false: alpha.metal convention
    scalar  solidusTemp_;
    scalar  searchRadius_;
    scalar  overlapThreshold_;
    scalar  exclusionScale_;
    vector  vLaser_;
    vector  laserOrigin_;
    bool    runPythonAtEnd_;
    fileName pythonScript_;

    // ---- persistent state ----
    std::map<label, PoreData> activePores_;
    label   nextPoreID_;        // next available persistent tracking ID (keyhole = 0)
    label   executionCount_;
    List<label> prevTrackIDPerCell_; // previous step persistent IDs per local cell

    // ---- output ----
    mutable std::unique_ptr<std::ofstream> csvPtr_;

    // ---- phase 1: local CCL ----
    List<label> localCCL
    (
        const scalarField& alpha,
        const scalarField& T
    ) const;

    // ---- phase 2: parallel stitching ----
    // Fills globalCompID with cross-processor globally unique component labels.
    void resolveParallelBoundaries
    (
        const List<label>& localCompID,
        List<label>&        globalCompID
    ) const;

    // ---- phase 3: pore properties (volume, centroid, force) ----
    std::map<label, PoreData> computePoreProperties
    (
        const List<label>&  globalCompID,
        const vectorField&  fLorentz
    ) const;

    // ---- phase 4: ID matching across time steps ----
    void matchAndUpdatePores
    (
        std::map<label, PoreData>& newPores,
        const std::map<label, std::map<label, label>>& overlapCounts
    );

    // nearest-pore lookup (within searchRadius_)
    label findNearestActivePore
    (
        const vector& pos,
        const std::set<label>& excluded
    ) const;

    // ---- output helpers ----
    void openCSV();
    void writeCSVRow(scalar t, const PoreData& p) const;

    // no copy
    PoreExtract(const PoreExtract&) = delete;
    void operator=(const PoreExtract&) = delete;

public:

    TypeName("PoreTracker");

    PoreExtract
    (
        const word&       name,
        const Time&       runTime,
        const dictionary& dict
    );

    virtual ~PoreExtract();

    virtual bool read(const dictionary& dict);
    virtual bool execute();
    virtual bool write();
    virtual bool end();
};


} // namespace functionObjects
} // namespace Foam

#endif
