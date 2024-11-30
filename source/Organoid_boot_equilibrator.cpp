#include <iostream>
#include <iomanip>
#include <math.h>
#include <tgmath.h> 
#include <unistd.h>
// #include <array>
#include <vector>
// #include <cstdlib>
// #include <ctime>
// #include <utility>
// #include <tuple>
// #include <cmath>
// #include <map>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <cstdlib>
#include <stdio.h>
// #include <cstring>
#include <string>
#include <random>
#include <chrono>
#include <sys/stat.h>
// #include <filesystem>
#include <string.h>
#include <algorithm>
#include <map>
#include <set>
#include <utility> // for std::pair
#include <variant>


///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////#DEFINITIONS///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
#define PI              (3.14159265359)
#define CYCLING_STATE   (1)
#define G1_ARR_STATE    (-1)
#define G0_STATE        (-2)
#define DIFF_STATE      (-3)
#define APOP_STATE      (-4)
#define CA_CELL_TYPE    (1)
#define WT_CELL_TYPE    (0)
#define NULL_CELL_TYPE  (-1) // Does not exist (index > NCells)
///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////#DEFINITIONS///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////////////////////////////// USING /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace std::chrono;
//////////////////////////////////////////////////////////////////////////
////////////////////////////// USING /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////////////////////////////// STRUCTS////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
struct Config {
    
    //############### GENERAL ##############
    int N_UpperLim; // upper limit for defining vectors
    int NTypes; // number of cell types.
    std::vector<double> typeR0; // R for each type @ phi = 0
    std::vector<double> typeR2PI; // R for each type @ phi = 2*PI
    std::vector<std::vector<double>> typeTypeEpsilon; // This is a matrix that shows synchronization strength of the cell phase for each pair of cell types which are neighbors;
    //############### GENERAL ##############

    //############### FORCE AND FRICTION ##############
    std::vector<double> typeGamma;
    std::vector<std::vector<double>> typeTypeGammaCC;
    std::vector<std::vector<double>> typeTypeF_rep_max;
    std::vector<std::vector<double>> typeTypeF_abs_max;
    double R_eq_coef;
    double R_cut_coef_force;
    //############### FORCE AND FRICTION ##############
    
    //############### SELF-PROPULSION ##############
    std::vector<double> typeFm;
    std::vector<double> typeDr;
    //############### SELF-PROPULSION ##############
    
    // //############### Phi Noise ##############
    // double PhiNoiseSigma;
    // //############### Phi Noise ##############

    //############### G1 phase border / (2*PI) ##############
    double G1Border;
    //############### G1 phase border / (2*PI) ##############
    
    // //############### THRESHOLD OMEGAS ##############
    // double omegaThG1_arr;
    // double omegaThG0;
    // double omegaThDiff;
    // double omegaThApop;
    // //############### THRESHOLD OMEGAS ##############

    //############### POTENTIAL LANDSCAPE ##############
    std::vector<double> typeOmega;
    std::vector<double> typeBarrierW;
    std::vector<double> typeSigmaPhi;
    std::vector<double> typeSigmaFit;
    double barrierPeakCoef;

    std::vector<double> typeFit0;
    double Fit_Th_Wall;
    double Fit_Th_G0;
    double Fit_Th_Diff;
    double Fit_Th_Apop;
    //############### POTENTIAL LANDSCAPE ##############
    
    //############### TIMING & SAMPLING ##############
    double maxTime;
    double dt;
    double dt_sample;
    int samplesPerWrite;
    int writePerZip;
    double printingTimeInterval;
    //############### TIMING & SAMPLING ##############

    // //############### FITNESS TO OMEGA MAPPING ##############
    // std::vector<double> typeOmega0;
    // std::vector<double> typeOmegaLim;
    // std::vector<double> typeFit0;
    // std::vector<double> typeFitLim;
    // //############### FITNESS TO OMEGA MAPPING ##############

    //############### GAME ##############
    double R_cut_coef_game;
    // std::vector<double> typeGameNoiseSigma;
    double tau;
    std::vector<std::vector<double>> typeTypePayOff_mat_real_C;
    std::vector<std::vector<double>> typeTypePayOff_mat_real_F1;
    std::vector<std::vector<double>> typeTypePayOff_mat_real_F2;
    std::vector<std::vector<double>> typeTypePayOff_mat_imag_C;
    std::vector<std::vector<double>> typeTypePayOff_mat_imag_F1;
    std::vector<std::vector<double>> typeTypePayOff_mat_imag_F2;
    //############### GAME ##############

    //############### NEWBORN FITNESS ##############
    std::string newBornFitKey;
    //############### NEWBORN FITNESS ##############

    //############### INITIALIZAION ##############
    std::string initConfig;
    //############### INITIALIZAION ##############

};

struct dist_data {

    std::vector<double> phi_tot = {};
    std::vector<double> fit_tot = {};
    std::vector<double> phi_G1 = {};
    std::vector<double> fit_G1 = {};
    std::vector<double> phi_G0 = {};
    std::vector<double> fit_G0 = {};
    std::vector<double> phi_G2 = {};
    std::vector<double> fit_G2 = {};

    
};

enum class ObjectType {
    VECTOR,
    MATRIX
};
//////////////////////////////////////////////////////////////////////////
////////////////////////////// STRUCTS////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////////////////////////////// PROTOTYPES ////////////////////////////////
//////////////////////////////////////////////////////////////////////////
dist_data dist_data_maker(const int NCells, const vector<int>& cellType,  const vector<double>& cellPhi, const vector<vector<double>>& cellFitness, \
                          const vector<int>& cellState, const Config config, const int Type);

int vector_condition_checker(const vector<double> VEC, const int numbers_for_check);

void simulationDataReader(int* NSitesPtr, double* LxPtr, double* LyPtr, double* AlphaPtr, double* KbPtr, double* TemPtr,  int* NumCellsPtr, \
                          double* AvgCellAreaPtr, double* LambdaPtr, long* maxMCStepsPtr, int* samplesPerWritePtr, \
                          int* printingTimeIntervalPtr, int* numLinkedListPtr, string* initConfigPtr);

std::vector<double> parseVector(const std::string& str);

std::vector<std::vector<double>> parseMatrix(const std::string& str);

void trim(std::string& str);

void readConfigFile(const std::string& filename, Config& config);

void initializer(const int N_UpperLim, int* NCellsPtr, vector<int>& NCellsPerType,
                 const vector<double> typeR0, const vector<double> typeR2PI, 
                 vector<int>& cellType, vector<double>& cellX, vector<double>& cellY, 
                 vector<double>& cellVx, vector<double>& cellVy, 
                 vector<double>& cellPhi, vector<int>& cellState, vector<double>& cellTheta,
                 vector<vector<double>>& cellFitness, const vector<double>& typeFit0, std::mt19937 &mt_rand);

void initial_read(const int N_UpperLim, int* NCellsPtr, vector<int>& NCellsPerType,
                 const vector<double> typeR0, const vector<double> typeR2PI, 
                 vector<int>& cellType, vector<double>& cellX, vector<double>& cellY, 
                 vector<double>& cellVx, vector<double>& cellVy, 
                 vector<double>& cellPhi, vector<int>& cellState, vector<double>& cellTheta,
                 vector<vector<double>>& cellFitness);

void R_Area_calc(const int N_UpperLim, const int NCells, const int NTypes,
                 const vector<double> typeR0, const vector<double> typeR2PI, 
                 const vector<int>& cellType,
                 const vector<double>& cellPhi, const vector<int>& cellState,
                 vector<double>& cellR, vector<double>& cellArea);

void writeIntVectorToFile(const std::vector<int>& vec, int NCells, const std::string& filename);

void writeIntMatrixToFile(const std::vector<std::vector<int>>& mat, const int N_rows_desired, const int N_cols_desired, const std::string& filename);

void writeDoubleVectorToFile(const std::vector<double>& vec, int NCells, const std::string& filename);

void writeDoubleMatrixToFile(const std::vector<std::vector<double>>& mat, const int N_rows_desired, const int N_cols_desired, const std::string& filename);

void readIntVectorFromFile(const std::string& filename, std::vector<int>& data);

void readDoubleVectorFromFile(const std::string& filename, std::vector<double>& data);

void readIntMatrixFromFile(const std::string& filename, std::vector<std::vector<int>>& data);

void readDoubleMatrixFromFile(const std::string& filename, std::vector<std::vector<double>>& data);

void dataBunchWriter(const int NCells, \
                     const vector<double> tBunch, \
                     const vector<vector<int>> cellTypeBunch, \
                     const vector<vector<double>> cellXBunch, \
                     const vector<vector<double>> cellYBunch, \
                     const vector<vector<double>> cellVxBunch, \
                     const vector<vector<double>> cellVyBunch, \
                     const vector<vector<double>> cellPhiBunch, \
                     const vector<vector<int>> cellStateBunch, \
                     const vector<vector<vector<double>>> cellFitnessBunch, \
                     const int saved_bunch_index);

std::vector<std::vector<int>> IntTranspose(const std::vector<std::vector<int>>& matrix);

std::vector<std::vector<double>> DoubleTranspose(const std::vector<std::vector<double>>& matrix);

void writeFitnessToFile(const std::vector<std::vector<std::vector<double>>>& matrix, const int N_rows_desired, const int N_cols_desired, const std::string& filename);

void writeVecOfVecsUnequal(const std::vector<std::vector<double>>& vec_of_vecs, const int N_desired, const std::string& filename);
//////////////////////////////////////////////////////////////////////////
////////////////////////////// PROTOTYPES ////////////////////////////////
//////////////////////////////////////////////////////////////////////////

int main()
{

    /////////////////// FOLDERS NAMES ////////////////////
    std::string dataFolderName = "data";
    std::string initFolderName = "init";
    std::string mainResumeFolderName = "main_resume";
    std::string backupResumeFolderName = "backup_resume";
    std::string loadFolderName;
    /////////////////// FOLDERS NAMES ////////////////////

    /////////////////// MAKING SUB DIRECTORIES /////////////////
    // This block is for windows:
    // mkdir(dataFolderName.c_str()); //making data folder
    // mkdir(initFolderName.c_str()); //making init folder
    // mkdir(mainResumeFolderName.c_str()); //making main_resume folder
    // mkdir(backupResumeFolderName.c_str()); //making backup_resume folder

    // This block is for Linux:
    // mkdir(dataFolderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making data folder
    // mkdir(initFolderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making init folder
    // mkdir(mainResumeFolderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making main_resume folder
    // mkdir(backupResumeFolderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making backup_resume folder
    /////////////////// MAKING SUB DIRECTORIES /////////////////


    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///////////// Parameters reading /////////////////////////////////////////////////////////////
    Config config;
    // readConfigFile("params_test.csv", config);
    readConfigFile("params.csv", config);

    ///////////////////////////////// VARIABLES DEFINITION /////////////////////////////
    //############### GENERAL ##############
    int N_UpperLim; // upper limit for defining vectors
    int NTypes; // number of cell types.
    vector<double> typeR0(NTypes); // The radius of each type at \phi = 0.0; 
    vector<double> typeR2PI(NTypes); // The radius of each type at \phi = 2*PI; 
    vector<vector<double>> typeTypeEpsilon(NTypes, vector<double>(NTypes)); // This is a matrix that shows synchronization strength of the cell phase for each pair of cell types which are neighbors;
    //############### GENERAL ##############

    //############### FORCE AND FRICTION ##############
    vector<double> typeGamma(NTypes); // velocity = (1/gamma) * total_force;
    vector<vector<double>> typeTypeGammaCC(NTypes, vector<double>(NTypes)); // This is a matrix that shows friction coefficient between each pair of cell types;
    vector<vector<double>> typeTypeF_rep_max(NTypes, vector<double>(NTypes)); // This is a matrix that shows F_rep_max between each pair of cell types;
    vector<vector<double>> typeTypeF_abs_max(NTypes, vector<double>(NTypes)); // This is a matrix that shows F_abs_max between each pair of cell types;
    double R_eq_coef; // R_eq_coef = (R_eq)/ (R[cell_1]+R[cell_2]);
    double R_cut_coef_force; //  R_cut_coef_force = (R_cut_off_force)/ (R[cell_1]+R[cell_2]); 
    //############### FORCE AND FRICTION ##############
    
    //############### SELF-PROPULSION ##############
    vector<double> typeFm(NTypes); // self-propulsion force amplitude of each cell type;
    vector<double> typeDr(NTypes); // Diffusion coef of self-propulsion direction of each cell type;
    //############### SELF-PROPULSION ##############

    //############### G1 phase border / (2*PI) ##############
    double G1Border; // The fraction of the whole period at which G1 ends (G1Border = (G1-end phase)/(2*PI) )
    //############### G1 phase border / (2*PI) ##############
    
    //############### POTENTIAL LANDSCAPE ##############
    vector<double> typeOmega(NTypes); // Omega for dPhi/dt
    vector<double> typeBarrierW(NTypes); // Width of Barrier; the std of the gaussian barrier
    vector<double> typeSigmaPhi(NTypes); // std of the Brownian noise in Phi
    vector<double> typeSigmaFit(NTypes); // std of the Brownian noise in Fitness
    double barrierPeakCoef;

    vector<double> typeFit0(NTypes); // F_0^WT and F_0^C, which are exactly omega_0 values.
    // double Fit_Th_G1_arr; // the fitness where WT cells go into G1-arrest. (only happens when they are already in G1).
    double Fit_Th_Wall; // where the infinite potential wall begins
    double Fit_Th_G0; // the fitness where WT cells go from G1-arrest into G0.
    double Fit_Th_Diff; // the fitness where WT cells go from G0 into differentiated state.
    double Fit_Th_Apop; // the fitness where WT cells go from differentiated state into apoptosis.
    //############### POTENTIAL LANDSCAPE ##############
    
    //############### TIMING & SAMPLING ##############
    double maxTime; // total simulaion time
    double dt; // simulation and integration time step
    double dt_sample; // sampling intervals
    int samplesPerWrite; // how many samples per each data writing operation
    int writePerZip; // how many written set of data for each zipping operation
    double printingTimeInterval; // for terminal show 
    //############### TIMING & SAMPLING ##############

    //############### GAME ##############
    double R_cut_coef_game;   //R_cut_coef_game = (R_cut_game)/ (R[cell_1]+R[cell_2]); 
    // vector<double>  typeGameNoiseSigma(NTypes); // std of the noise term in payoff matrix entries
    double tau; // the characteristic memory time for cells fitnesses
    vector<vector<double>> typeTypePayOff_mat_real_C(NTypes, vector<double>(NTypes)); // constant term
    vector<vector<double>> typeTypePayOff_mat_real_F1(NTypes, vector<double>(NTypes)); // coefficient of fitness of player no.1 (the one that GAINS the value in the payoff matrix)
    vector<vector<double>> typeTypePayOff_mat_real_F2(NTypes, vector<double>(NTypes)); // coefficient of fitness of player no.2 (the one that LOSES the value in the payoff matrix)
    vector<vector<double>> typeTypePayOff_mat_imag_C(NTypes, vector<double>(NTypes)); // imaginary values like above
    vector<vector<double>> typeTypePayOff_mat_imag_F1(NTypes, vector<double>(NTypes));
    vector<vector<double>> typeTypePayOff_mat_imag_F2(NTypes, vector<double>(NTypes));
    //############### GAME ##############

    //############### NEWBORN FITNESS ##############
    std::string newBornFitKey;
    //############### NEWBORN FITNESS ##############

    //############### INITIALIZAION ##############
    std::string initConfig;
    //############### INITIALIZAION ##############
    ///////////////////////////////// VARIABLES DEFINITION /////////////////////////////


    ///////////////////////////////// VARIABLES VALUE ASSIGHNMENT /////////////////////
    //############### GENERAL ##############
    N_UpperLim = config.N_UpperLim; // upper limit for defining vectors
    NTypes = config.NTypes; // number of cell types.
    typeR0 = config.typeR0; // The radius of each type at \phi = 0.0; 
    typeR2PI = config.typeR2PI; // The radius of each type at \phi = 2*PI; 
    typeTypeEpsilon = config.typeTypeEpsilon; // This is a matrix that shows synchronization strength of the cell phase for each pair of cell types which are neighbors;
    //############### GENERAL ##############

    //############### FORCE AND FRICTION ##############
    typeGamma = config.typeGamma; // velocity = (1/gamma) * total_force;
    typeTypeGammaCC = config.typeTypeGammaCC; // This is a matrix that shows friction coefficient between each pair of cell types;
    typeTypeF_rep_max = config.typeTypeF_rep_max; // This is a matrix that shows F_rep_max between each pair of cell types;
    typeTypeF_abs_max = config.typeTypeF_abs_max; // This is a matrix that shows F_abs_max between each pair of cell types;
    R_eq_coef = config.R_eq_coef; // R_eq_coef = (R_eq)/ (R[cell_1]+R[cell_2]);
    R_cut_coef_force = config.R_cut_coef_force; //  R_cut_coef_force = (R_cut_off_force)/ (R[cell_1]+R[cell_2]); 
    //############### FORCE AND FRICTION ##############
    
    //############### SELF-PROPULSION ##############
    typeFm = config.typeFm; // self-propulsion force amplitude of each cell type;
    typeDr = config.typeDr; // Diffusion coef of self-propulsion direction of each cell type;
    //############### SELF-PROPULSION ##############

    //############### G1 phase border / (2*PI) ##############
    G1Border = config.G1Border; // The fraction of the whole period at which G1 ends (G1Border = (G1-end phase)/(2*PI) )
    //############### G1 phase border / (2*PI) ##############
    
    //############### POTENTIAL LANDSCAPE ##############
    typeOmega = config.typeOmega;
    typeBarrierW = config.typeBarrierW;
    typeSigmaPhi = config.typeSigmaPhi;
    typeSigmaFit = config.typeSigmaFit;
    barrierPeakCoef = config.barrierPeakCoef;

    typeFit0 = config.typeFit0; // inherent fitnesses
    // Fit_Th_G1_arr = config.Fit_Th_G1_arr; // the fitness where WT cells go into G1-arrest. (only happens when they are already in G1).
    Fit_Th_Wall = config.Fit_Th_Wall;
    Fit_Th_G0 = config.Fit_Th_G0; // the fitness where WT cells go from G1-arrest into G0.
    Fit_Th_Diff = config.Fit_Th_Diff; // the fitness where WT cells go from G0 into differentiated state.
    Fit_Th_Apop = config.Fit_Th_Apop; // the fitness where WT cells go from differentiated state into apoptosis.
    double fit_eps = (1e-6) * abs(typeFit0[0] - Fit_Th_G0);
    double phi_eps = (1e-8) * (2.0 * PI);
    //############### POTENTIAL LANDSCAPE ##############
    
    //############### TIMING & SAMPLING ##############
    maxTime = config.maxTime; // total simulaion time
    dt = config.dt; // simulation and integration time step
    dt_sample = config.dt_sample; // sampling intervals
    samplesPerWrite = config.samplesPerWrite; // how many samples per each data writing operation
    writePerZip = config.writePerZip; // how many written set of data for each zipping operation
    printingTimeInterval = config.printingTimeInterval; // for terminal show 
    //############### TIMING & SAMPLING ##############

    //############### GAME ##############
    R_cut_coef_game = config.R_cut_coef_game;   //R_cut_coef_game = (R_cut_game)/ (R[cell_1]+R[cell_2]); 
    // typeGameNoiseSigma = config.typeGameNoiseSigma; // std of the noise term in payoff matrix entries
    tau = config.tau;
    typeTypePayOff_mat_real_C = config.typeTypePayOff_mat_real_C; // constant term
    typeTypePayOff_mat_real_F1 = config.typeTypePayOff_mat_real_F1; // coefficient of fitness of player no.1 (the one that GAINS the value in the payoff matrix)
    typeTypePayOff_mat_real_F2 = config.typeTypePayOff_mat_real_F2; // coefficient of fitness of player no.2 (the one that LOSES the value in the payoff matrix)
    typeTypePayOff_mat_imag_C = config.typeTypePayOff_mat_imag_C; // imaginary values like above
    typeTypePayOff_mat_imag_F1 = config.typeTypePayOff_mat_imag_F1;
    typeTypePayOff_mat_imag_F2 = config.typeTypePayOff_mat_imag_F2;
    //############### GAME ##############

    //############### NEWBORN FITNESS ##############
    newBornFitKey =  config.newBornFitKey;
    //############### NEWBORN FITNESS ##############

    //############### INITIALIZAION ##############
    initConfig =  config.initConfig;
    //############### INITIALIZAION ##############

    double deltaFit_computational_cut = barrierPeakCoef / (typeOmega[1]*2*PI*G1Border); // This means that at the cutoff fitness value, the height is considered to be equal to \phi=0;
    ///////////////////////////////// VARIABLES VALUE ASSIGHNMENT /////////////////////


    double motherFitnessWeight, inherentFitnessWeight;
    if (newBornFitKey == "FI") // full interitence
    {
        motherFitnessWeight = 1.0;
    } else if (newBornFitKey == "r") // full reset
    {
        motherFitnessWeight = 0.0;
    }
    else if (newBornFitKey == "ec") // economic fitness inheritence
    {
        motherFitnessWeight = 0.5;
    }
    else
    {
        std::ofstream errorFile("error.log");
        errorFile << "Error: newBornFitKey in params.csv has a wrong vlalue!" << std::endl;
        errorFile.close();
        return 0;
    }
    inherentFitnessWeight = 1.0 - motherFitnessWeight;
    


    vector<double> typeA_min(NTypes); // minimum area of each cell type
    vector<double> typeA_max(NTypes); // maximum area of each cell type
    for (int type_c = 0; type_c < NTypes; type_c++)
    {
        typeA_min[type_c] = PI * typeR0[type_c] * typeR0[type_c];
        typeA_max[type_c] = PI * typeR2PI[type_c] * typeR2PI[type_c];
    }
    ///////////// Parameters reading /////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////

    


    int NCells;     // total number of cells (changes during the simulation)
    int NCells_orig;
    vector<int> NCellsPerType(NTypes);
    vector<int> cellType(N_UpperLim);
    
    vector<double> cellX(N_UpperLim);
    vector<double> cellY(N_UpperLim);
    vector<double> cellVx(N_UpperLim);
    vector<double> cellVy(N_UpperLim);
    vector<double> cellFx(N_UpperLim); // force x
    vector<double> cellFy(N_UpperLim); // force y
    // vector<double> cellSigmaGammaCC(N_UpperLim); // The sum of \gamma_{cc} with neighbors, for each cell
    vector<double> cellTheta(N_UpperLim); // self-propulsion direction of cells, \in [0, 2*pi]

    vector<double> cellJ(N_UpperLim); // input flux of fitness
    vector<double> cellPhi(N_UpperLim); // phase in cell cycle, \in [0, 2*pi]
    vector<int> cellState(N_UpperLim); // Cell state : {cycling:CYCLING_STATE, G1_arr:G1_ARR_STATE, G0:G0_STATE, differentiated:DIFF_STATE, apop:APOP_STATE, does not exist: 0}
    // vector<double> cellOmega(N_UpperLim); // cell fitness. Changes by the game. equals to d phi / dt for cycling cells
    vector<double> cellSync(N_UpperLim); // Kuramoto synchronization term for each cell
    
    
    vector<double> cellR(N_UpperLim); // The radius of each cell. It may change time to time.
    vector<double> cellArea(N_UpperLim); // The area of each cell. It may change time to time.
    vector<vector<double>> cellFitness(N_UpperLim, vector<double>(2)); // This stores the complex fitness of cells (Real and Imaginary parts).

    vector<vector<double>> divisionTimes(N_UpperLim); // the times at which cells divide

    ///////////// BUNCHES FOR WRITING SAMPLES /////////////////
    vector<double> tBunch;
    vector<vector<int>> cellTypeBunch;
    vector<vector<double>> cellXBunch;
    vector<vector<double>> cellYBunch;
    vector<vector<double>> cellVxBunch;
    vector<vector<double>> cellVyBunch;
    vector<vector<double>> cellPhiBunch;
    vector<vector<int>> cellStateBunch;
    vector<vector<double>> cellRBunch;
    vector<vector<vector<double>>> cellFitnessBunch;
    int writeCounter = 0;
    ///////////// BUNCHES FOR WRITING SAMPLES /////////////////

    ////////////////////// DEFINING RANDOM GENERATOR ///////////////
    std::mt19937 mt_rand;
    unsigned long randState;
    ////////////////////// DEFINING RANDOM GENERATOR ///////////////

    /////////////////// RANDOM GENERATOR SEEDING /////////////////
    random_device rd; // random seed creation
    unsigned long mt_rand_seed = rd();
    // mt_rand_seed = 2797319747;

    //Seeding
    mt_rand.seed(mt_rand_seed);
    unsigned long MT_MAX = mt_rand.max();
    unsigned long MT_MIN = mt_rand.min();
    // saving initial random seed
    // ofstream randSeedInit;
    // randSeedInit.open(initFolderName + "/" + "randSeedInit.csv");
    // randSeedInit << mt_rand_seed;
    // randSeedInit.close(); // random seed saved
    

    // // saving initial random generator
    // std::ofstream randStateInit(initFolderName + "/" + "randStateInit.csv");
    // randStateInit << mt_rand;
    // randStateInit.close();
    /////////////////// RANDOM GENERATOR SEEDING /////////////////
    
    
    ///////////////// INITIALIZATION ////////////////////
    int saved_bunch_index = 1;
    if ( initConfig == "read" )
    {
    initial_read(N_UpperLim, &NCells, NCellsPerType,
                typeR0, typeR2PI, 
                cellType, cellX, cellY, 
                cellVx, cellVy, 
                cellPhi, cellState, cellTheta,
                cellFitness);
    } else if ( initConfig == "gen" )
    {

    std::ofstream errorFile("error.log");
    errorFile << "Error: the 'gen' mode not allowed!" << std::endl;
    errorFile.close();

    return 1;

    // initializer(N_UpperLim, &NCells, NCellsPerType,
    //              typeR0, typeR2PI, 
    //              cellType, cellX, cellY, 
    //              cellVx, cellVy, 
    //              cellPhi, cellState, cellTheta,
    //              cellFitness, typeFit0, mt_rand);
    }
    R_Area_calc(N_UpperLim, NCells, NTypes,
                typeR0, typeR2PI, 
                cellType,
                cellPhi, cellState,
                cellR, cellArea); // cellR and cellArea are calculated. The others are const in the function.
    
    NCells_orig = NCells;
    ///////////////// INITIALIZATION ////////////////////



    // double dt = 0.01;
    double t = dt;
    double sqrt_dt = pow(dt, 0.5);
    double t_eps = 0.01 * dt;
    double tLastSampling = 0.0;
    int bunchInd = 1;
    double Fx, Fy, F; // these are forces
    double J_input_real_1, J_input_imag_1, J_input_real_2, J_input_imag_2;
    int cellC_1, cellC_2;
    double distance, distance2; // distance2 means distance^2
    double delta_x, delta_y; // that of 2 minus that of 1
    double R_cut_force,R_cut_game, R_eq, R_peak;
    int cellType_1, cellType_2; // type of cell 1 and type of cell 2
    double deltaOmega, KuramotoTerm;
    double FaddTermX, FaddTermY, gammaCC_val;
    int newBornCells, newBornInd, dyingCells;
    // double A_min, A_max;
    double cellAreaUpdate_val;
    double rand_divison_angle;
    double daughterX_1, daughterY_1, daughterX_2, daughterY_2;
    double max_interaction_r2, max_interaction_r; // maximum interaction distance, and its square.
    max_interaction_r =  (max(R_cut_coef_force, R_cut_coef_game)) * 2.0 * (*std::max_element(typeR2PI.begin(), typeR2PI.end()));
    max_interaction_r2 = max_interaction_r * max_interaction_r;
    double fitness_noise_real, fitness_noise_imag;
    unsigned long long int_rand_1, int_rand_2;
    double uniform_rand_1, uniform_rand_2, gauss_rand_1, gauss_rand_2; // for Box-Muller transform
    double SyncTerm;
    double phiNoise, fitNoise, peakFactor, deltaPhi, phiMid, funcOld, funcNew, barrierD, funcBarrier;
    // k_0 = 1.0 / (1.0e-5);

    // for identifying immediate neighbors
    // int NN_force[N_UpperLim][N_UpperLim];
    // int NN_game[N_UpperLim][N_UpperLim];
    vector<vector<int>> NN_force(N_UpperLim, vector<int>(N_UpperLim));
    vector<vector<int>> NN_game(N_UpperLim, vector<int>(N_UpperLim));
    
    
    
    
    // Update Auxiliary properties vectors
    vector<vector<double>> cellFitnessUpdated(N_UpperLim, vector<double>(2)); // This stores the updated values of fitness of cells (Real and Imaginary parts).
    vector<double> cellPhiUpdated(N_UpperLim); // Updated phases in cell cycle, \in [0, 2*pi]
    vector<double> cellRUpdated(N_UpperLim); // Updated Radius of the cells
    vector<int> cellStateUpdated(N_UpperLim); // Cell state Updated values: {cycling:CYCLING_STATE, G1_arr:G1_ARR_STATE, G0:G0_STATE, differentiated:DIFF_STATE, apop:APOP_STATE, does not exist: 0}

    vector<double> cellXUpdated(N_UpperLim); // Updated X
    vector<double> cellYUpdated(N_UpperLim); // Updated Y

    
    // Update Auxiliary properties vectors

    int debug_var;

    int equilibration_cond = 0;
    int sampling_step = 0;

    
    // // these are vectors for saving values in each step of sampling
    // vector<double> avg_fit_G1;
    // vector<double> avg_phi_G1;
    // vector<double> avg_fit_G2;
    // vector<double> avg_phi_G2;
    // vector<double> avg_fit_G0;
    // vector<double> avg_phi_G0;

    int numbers_for_check = 4;
    std::vector<vector<dist_data>> distributions_steps(NTypes, vector<dist_data>());

    // loop of the sampling steps
    while (equilibration_cond == 0)
    {
        // initial zero-making
        for (cellC_1 = 0; cellC_1 < N_UpperLim; cellC_1++)
        {
            for (cellC_2 = 0; cellC_2 < N_UpperLim; cellC_2++)
            {
                NN_force[cellC_1][cellC_2] = 0;
                NN_game[cellC_1][cellC_2] = 0;
            }
        }

        for (cellC_1 = 0; cellC_1 < N_UpperLim; cellC_1++) // initializing: setting everything to zero
        {
            cellFx[cellC_1] = 0.0;
            cellFy[cellC_1] = 0.0;
            // cellSigmaGammaCC[cellC_1] = 0.0;

            cellJ[cellC_1] = 0.0;

            cellFitnessUpdated[cellC_1][0] = 0.0;
            cellFitnessUpdated[cellC_1][1] = 0.0;

            cellPhiUpdated[cellC_1] = 0.0;
            cellRUpdated[cellC_1] = 0.0;

            // cellStateUpdated[cellC_1] = 0;

            cellXUpdated[cellC_1] = 0.0;
            cellYUpdated[cellC_1] = 0.0;

            cellSync[cellC_1] = 0.0;
        }
        // initial zero-making

        // simulation loop comes here
        /////// SIMULATION LOOP /////////////
        while (t < maxTime)
        {

            // This loop for: Non-interactive updates, zero assighnment of interactive mediate tools.
            for (cellC_1 = 0; cellC_1 < NCells; cellC_1++)
            {
                // cellXUpdateed, cellYUpdated
                cellXUpdated[cellC_1] = cellX[cellC_1] + dt * cellVx[cellC_1];
                cellYUpdated[cellC_1] = cellY[cellC_1] + dt * cellVy[cellC_1];
                // cellXUpdateed, cellYUpdated

                cellFx[cellC_1]= 0.0;
                cellFy[cellC_1]= 0.0;
                // cellSigmaGammaCC[cellC_1] = 0.0;

                cellJ[cellC_1] = 0.0;
                cellSync[cellC_1] = 0.0;

            }
            // setting forces, and fluxes to zero
            
            // calculating Fx , Fy, and Updated fitnesses, without changing X, Y, Vx, Vy, and fitnesses
            for (cellC_1 = 0; cellC_1 < NCells; cellC_1++) // loop on cellC_1
            {

                if (cellState[cellC_1] == APOP_STATE)
                {
                    cellPhiUpdated[cellC_1] = cellPhi[cellC_1];
                    cellFitnessUpdated[cellC_1][0] = cellFitness[cellC_1][0];
                    cellRUpdated[cellC_1] = 0.0;

                    continue; // if the cell is already dead, there is nothing to do. Go to the next.
                }
                
                cellType_1 = cellType[cellC_1];

                // cellOmega[cellC_1] = typeOmega0[cellType_1];
                // cellOmega[cellC_1] = typeFit0[cellType_1];
                // KuramotoTerm = 0.0;

                for (cellC_2 = cellC_1 + 1 ; cellC_2 < NCells; cellC_2++) // loop on cellC_2, for interactions (force and game)
                {

                    if (cellState[cellC_2] == APOP_STATE)
                    {
                        continue; // if the cell is already dead, there is nothing to do. Go to the next.
                    }
                    
                    delta_x = cellX[cellC_2] - cellX[cellC_1];
                    delta_y = cellY[cellC_2] - cellY[cellC_1];
                    distance2 = delta_x * delta_x + delta_y * delta_y;

                    if (distance2 > max_interaction_r2) // if the distance is larger than the maximum interaction distance, do nothing and go to the next cellC_2. No mutual interaction!
                    {
                        NN_force[cellC_1][cellC_2] = 0;
                        NN_force[cellC_2][cellC_1] = 0;
                        NN_game[cellC_1][cellC_2] = 0;
                        NN_game[cellC_2][cellC_1] = 0;
                    }
                    else // if (distance2 <= max_interaction_r2), they MAY interact. 
                    {
                        distance = pow(distance2, 0.5);

                        R_cut_force = R_cut_coef_force * (cellR[cellC_2] + cellR[cellC_1]);

                        cellType_2 = cellType[cellC_2];

                        /////////// Cell-cell friction Forces ////////////////
                        if (distance < R_cut_force ) // They do interact forcewise
                        {
                            
                            NN_force[cellC_1][cellC_2] = 1;
                            NN_force[cellC_2][cellC_1] = 1;


                            FaddTermX = typeTypeGammaCC[cellType_1][cellType_2] * (cellVx[cellC_2] - cellVx[cellC_1]) ;
                            FaddTermY = typeTypeGammaCC[cellType_1][cellType_2] * (cellVy[cellC_2] - cellVy[cellC_1]) ;


                            cellFx[cellC_1] += FaddTermX;
                            cellFy[cellC_1] += FaddTermY;

                            cellFx[cellC_2] -= FaddTermX;
                            cellFy[cellC_2] -= FaddTermY;
                        }
                        else
                        {
                            NN_force[cellC_1][cellC_2] = 0;
                            NN_force[cellC_2][cellC_1] = 0;
                        }
                        /////////// Cell-cell friction Forces ////////////////

                        /////////// Game- and Sync interactions ////////////////
                        R_cut_game = R_cut_coef_game * (cellR[cellC_2] + cellR[cellC_1]);
                        if (distance < R_cut_game ) // They do interact gamewise
                        {   
                            //
                            NN_game[cellC_1][cellC_2] = 1;
                            NN_game[cellC_2][cellC_1] = 1;
                            //

                            // Kuramoto
                            SyncTerm = (typeTypeEpsilon[cellType_1][cellType_2] * sin( (cellPhi[cellC_2] - cellPhi[cellC_1])/2.0 ) );
                            cellSync[cellC_1] += SyncTerm;
                            cellSync[cellC_2] -= SyncTerm;

                            J_input_real_1     = typeTypePayOff_mat_real_C[cellType_1][cellType_2] + \
                                                typeTypePayOff_mat_real_F1[cellType_1][cellType_2] * cellFitness[cellC_1][0] + \
                                                typeTypePayOff_mat_real_F2[cellType_1][cellType_2] * cellFitness[cellC_2][0]; // This flux goes INTO cellC_1

                            J_input_real_2     = typeTypePayOff_mat_real_C[cellType_2][cellType_1] + \
                                                typeTypePayOff_mat_real_F1[cellType_2][cellType_1] * cellFitness[cellC_2][0] + \
                                                typeTypePayOff_mat_real_F2[cellType_2][cellType_1] * cellFitness[cellC_1][0]; // This flux goes INTO cellC_2
                            
                            cellJ[cellC_1] += J_input_real_1; // This flux goes INTO cellC_1
                            cellJ[cellC_2] += J_input_real_2; // This flux goes INTO cellC_2 


                        } // end of "if (distance < R_cut_game )"
                        else
                        {
                            NN_game[cellC_1][cellC_2] = 0;
                            NN_game[cellC_2][cellC_1] = 0;
                        }
                        /////////// Game- and Sync interactions ////////////////
                    } // end of "if (distance2 > max_interaction_r2){} else {}" 
                } // end of "for (cellC_2 = cellC_1 + 1 ; cellC_2 < NCells; cellC_2++)"

                



                if (cellState[cellC_1] != APOP_STATE) // it should be alive
                {

                    // The Updated versions of Phi, R, and Fitness of cellC_1 are calculated here
                    // noises calc
                    int_rand_1 = mt_rand();
                    while(int_rand_1 == MT_MIN || int_rand_1 == MT_MAX){int_rand_1 = mt_rand();}
                    int_rand_2 = mt_rand();
                    while(int_rand_2 == MT_MIN || int_rand_2 == MT_MAX){int_rand_2 = mt_rand();}
                    uniform_rand_1 = ((long double)(int_rand_1)-MT_MIN)/((long double)MT_MAX-MT_MIN);
                    uniform_rand_2 = ((long double)(int_rand_2)-MT_MIN)/((long double)MT_MAX-MT_MIN);
                    gauss_rand_1 =  pow( (-2.0 * log(uniform_rand_1)) , 0.5) * cos(2.0 * PI * uniform_rand_2); // Box-Muller transform

                    int_rand_1 = mt_rand();
                    while(int_rand_1 == MT_MIN || int_rand_1 == MT_MAX){int_rand_1 = mt_rand();}
                    int_rand_2 = mt_rand();
                    while(int_rand_2 == MT_MIN || int_rand_2 == MT_MAX){int_rand_2 = mt_rand();}
                    uniform_rand_1 = ((long double)(int_rand_1)-MT_MIN)/((long double)MT_MAX-MT_MIN);
                    uniform_rand_2 = ((long double)(int_rand_2)-MT_MIN)/((long double)MT_MAX-MT_MIN);
                    gauss_rand_2 =  pow( (-2.0 * log(uniform_rand_1)) , 0.5) * cos(2.0 * PI * uniform_rand_2); // Box-Muller transform

                    phiNoise = typeSigmaPhi[cellType_1] * sqrt_dt * gauss_rand_1;
                    fitNoise = typeSigmaFit[cellType_1] * sqrt_dt * gauss_rand_2;
                    // noises calc

                    // barrierSTD = typeBarrierW[cellType_1]/2.0; // the width of barrier is twice as the std of it.

                    barrierD = typeBarrierW[cellType_1]/2.0; // the width of barrier = 2 * D.

                    if (cellFitness[cellC_1][0] > Fit_Th_Wall + deltaFit_computational_cut)
                    {
                        // peakFactor = (barrierPeakCoef / (cellFitness[cellC_1][0]-Fit_Th_Wall) ) / (2.506628 * pow(barrierSTD, 3)); // this was for gaussian
                        peakFactor = (barrierPeakCoef / (cellFitness[cellC_1][0]-Fit_Th_Wall) );
                        // 2.506628 = sqrt(2*PI)
                    } else
                    {
                        // peakFactor = (barrierPeakCoef / (fit_eps) ) / (2.506628 * pow(barrierSTD, 3));
                        peakFactor = (barrierPeakCoef / (deltaFit_computational_cut) );
                        // 2.506628 = sqrt(2*PI)
                    }
                    // if (peakFactor < 0)
                    // {
                    //     cout<<"######################"<<endl;
                    //     cout<<"peakFactor < 0"<<endl;
                    //     cout<<"cellFitness[cellC_1][0] = "<<cellFitness[cellC_1][0]<<endl;
                    //     cout<<"######################"<<endl;
                    // }
                    
                    
                    

                    // prediction (first step for phi)
                    deltaPhi = (cellPhi[cellC_1] - G1Border*2*PI);
                    // funcOld = (typeOmega[cellType_1] + peakFactor * deltaPhi * exp(- deltaPhi * deltaPhi / (2 * barrierSTD * barrierSTD)) + cellSync[cellC_1]);
                    funcBarrier = (abs(deltaPhi)<barrierD) * (0.5 * PI / barrierD) * peakFactor * sin(deltaPhi * PI / barrierD) * (1+cos(deltaPhi * PI / barrierD));
                    funcOld = (typeOmega[cellType_1] + funcBarrier + cellSync[cellC_1]);
                    phiMid = cellPhi[cellC_1] + \
                            dt * funcOld + \
                            phiNoise;
                    
                    // correction (second step for phi)
                    deltaPhi = (phiMid - G1Border*2*PI);
                    // funcNew = (typeOmega[cellType_1] + peakFactor * deltaPhi * exp(- deltaPhi * deltaPhi / (2 * barrierSTD * barrierSTD)) + cellSync[cellC_1]);
                    funcBarrier = (abs(deltaPhi)<barrierD) * (0.5 * PI / barrierD) * peakFactor * sin(deltaPhi * PI / barrierD) * (1+cos(deltaPhi * PI / barrierD));
                    funcNew = (typeOmega[cellType_1] + funcBarrier + cellSync[cellC_1]);
                    cellPhiUpdated[cellC_1] = cellPhi[cellC_1] + \
                                            dt * 0.5 * (funcOld + funcNew) + \
                                            phiNoise;
                    
                    // if (cellPhiUpdated[cellC_1] > G1Border*2*PI && cellPhi[cellC_1] < G1Border*2*PI)
                    // {
                    //     cout<<"######################"<<endl;
                    //     cout<<"cellFitness[cellC_1][0] = "<<cellFitness[cellC_1][0]<<endl;
                    //     cout<<"peakFactor = "<<peakFactor<<endl;
                    //     cout<<"######################"<<endl;
                    // }
                    

                    if (cellPhiUpdated[cellC_1] < 0)
                    {
                        cellPhiUpdated[cellC_1] = phi_eps;
                    }
                    // fitness update
                    // cellFitnessUpdated[cellC_1][0] = cellFitness[cellC_1][0] + \
                    //                                 fitNoise + \
                    //                                 dt * ( \
                    //                                             cellJ[cellC_1] + \
                    //                                             (-1.0 / tau) * (cellFitness[cellC_1][0] - typeFit0[cellType_1]) 
                    //                                         );
                    
                    cellFitnessUpdated[cellC_1][0] = cellFitness[cellC_1][0] + \
                                                    fitNoise + \
                                                    dt * cellJ[cellC_1];
                    
                    // if (   (abs(cellPhiUpdated[cellC_1] - G1Border*2*PI) < barrierD ) \
                    //     && (cellFitnessUpdated[cellC_1][0] < Fit_Th_Wall + deltaFit_computational_cut ) )
                    // {
                    //     cellFitnessUpdated[cellC_1][0] = cellFitness[cellC_1][0];
                    // }
                    

                }
                else // if it is dead, nothing updates
                {
                    cellPhiUpdated[cellC_1] = cellPhi[cellC_1];
                    cellFitnessUpdated[cellC_1][0] = cellFitness[cellC_1][0];
                }


                cellAreaUpdate_val = typeA_min[cellType_1] + (typeA_max[cellType_1] - typeA_min[cellType_1]) * cellPhiUpdated[cellC_1] / (2 * PI);
                cellRUpdated[cellC_1] = pow(cellAreaUpdate_val / PI, 0.5);
                // The Updated versions of Phi, R, and Fitness of cellC_1 are calculated here
                
            } // the end of "for (cellC_1 = 0; cellC_1 < NCells; cellC_1++)"



            // These two for loops are for calculating center-to-center force terms
            for (cellC_1 = 0; cellC_1 < NCells; cellC_1++) // loop on cellC_1
            {
                if (cellState[cellC_1] == APOP_STATE)
                {
                    cellVx[cellC_1] =   0.0 ;
                    cellVy[cellC_1] =   0.0 ;

                    continue; // if the cell is already dead, there is nothing to do. Go to the next.
                }

                cellType_1 = cellType[cellC_1];

                for (cellC_2 = cellC_1 + 1 ; cellC_2 < NCells; cellC_2++) // loop on cellC_2, for interactions (force and game)
                {
                    if (cellState[cellC_2] == APOP_STATE)
                    {
                        continue; // if the cell is already dead, there is nothing to do. Go to the next.
                    }

                    cellType_2 = cellType[cellC_2];

                    if (NN_force[cellC_1][cellC_2])
                    {
                        
                        delta_x = cellXUpdated[cellC_2] - cellXUpdated[cellC_1];
                        delta_y = cellYUpdated[cellC_2] - cellYUpdated[cellC_1];
                        distance = pow( (delta_x * delta_x + delta_y * delta_y) , 0.5);

                        R_eq =   R_eq_coef * (cellRUpdated[cellC_2] + cellRUpdated[cellC_1]);
                        R_cut_force = R_cut_coef_force * (cellRUpdated[cellC_2] + cellRUpdated[cellC_1]);

                        if (distance < R_eq )
                        {
                            F = typeTypeF_rep_max[cellType_1][cellType_2] * (distance - R_eq) / R_eq;
                        } else
                        {
                            // F = typeTypeF_abs_max[cellType_1][cellType_2] * (distance - R_eq) / (R_cut_force - R_eq);

                            // F = (-4.0 * typeTypeF_abs_max[cellType_1][cellType_2]/((R_cut_force-R_eq)*(R_cut_force-R_eq))) * \
                            //  (distance - R_eq) * (distance - R_cut_force);

                            R_peak = 0.5 * (R_eq + R_cut_force);

                            if (distance < R_peak)
                            {
                                F = typeTypeF_abs_max[cellType_1][cellType_2] * (distance - R_eq) / (R_peak - R_eq);
                            }
                            else
                            {
                                F = typeTypeF_abs_max[cellType_1][cellType_2] * ( 1.0 - (distance - R_peak) / (R_cut_force - R_peak) );
                            }
                            
                        }
                        

                        FaddTermX =  F * (delta_x / distance);
                        FaddTermY =  F * (delta_y / distance);

                        cellFx[cellC_1] += FaddTermX;
                        cellFy[cellC_1] += FaddTermY;

                        cellFx[cellC_2] -= FaddTermX;
                        cellFy[cellC_2] -= FaddTermY;

                    } // the end of "if (NN_force[cellC_1][cellC_2])"

                } // the end of "for (cellC_2 = cellC_1 + 1 ; cellC_2 < NCells; cellC_2++)"

                

                // Here, the V(t+dt) for cellC_1 is calculated
                cellVx[cellC_1] =   cellFx[cellC_1]  / typeGamma[cellType_1] ;
                cellVy[cellC_1] =   cellFy[cellC_1]  / typeGamma[cellType_1] ;
                // Here, the V(t+dt) for cellC_1 is calculated



            } // the end of "for (cellC_1 = 0; cellC_1 < NCells; cellC_1++)"
            // These two for loops are for calculating center-to-center force terms


            newBornCells = 0;
            newBornInd = NCells;
            dyingCells = 0;
            for (cellC_1 = 0; cellC_1 < NCells; cellC_1++)
            {
                cellType_1 = cellType[cellC_1];

                // Updating the original vectors; these also work for dead cells
                cellX[cellC_1] = cellXUpdated[cellC_1];
                cellY[cellC_1] = cellYUpdated[cellC_1];
                // Updating the original vectors; these also work for dead cells
                
                cellPhi[cellC_1] = cellPhiUpdated[cellC_1];
                cellR[cellC_1] = cellRUpdated[cellC_1];
                // cellFitness[cellC_1][0] = cellFitnessUpdated[cellC_1][0];
                // cellFitness[cellC_1][1] = cellFitnessUpdated[cellC_1][1];
                // Updating the original vectors


                // cell division
                if (cellPhi[cellC_1] >= 2.0 *PI)
                {   
                    cellFitness[cellC_1][0] = cellFitnessUpdated[cellC_1][0];

                    rand_divison_angle = (((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN)) * (2.0 * PI);

                    daughterX_1 = cellX[cellC_1] + (cellR[cellC_1] / 1.4142) * cos(rand_divison_angle);
                    daughterY_1 = cellY[cellC_1] + (cellR[cellC_1] / 1.4142) * sin(rand_divison_angle);
                    daughterX_2 = cellX[cellC_1] - (cellR[cellC_1] / 1.4142) * cos(rand_divison_angle);
                    daughterY_2 = cellY[cellC_1] - (cellR[cellC_1] / 1.4142) * sin(rand_divison_angle);
                    
                    // new Born cell
                    cellType[newBornInd] = cellType_1;

                    cellX[newBornInd] = daughterX_2;
                    cellY[newBornInd] = daughterY_2;

                    cellVx[newBornInd] = cellVx[cellC_1] / 2.0;
                    cellVy[newBornInd] = cellVy[cellC_1] / 2.0;

                    // cellArea[newBornInd] = A_min;
                    cellPhi[newBornInd] = phi_eps ;
                    cellR[newBornInd] = typeR0[cellType_1];
                    cellState[newBornInd] = cellState[cellC_1];

                    // cellFitness[newBornInd][0] = cellFitness[cellC_1][0];
                    cellFitness[newBornInd][0] = motherFitnessWeight * cellFitness[cellC_1][0] + inherentFitnessWeight * typeFit0[cellType_1];
                    // cellFitness[newBornInd][1] = cellFitness[cellC_1][1];
                    // new Born cell

                    // original cell
                    cellX[cellC_1] = daughterX_1;
                    cellY[cellC_1] = daughterY_1;

                    cellVx[cellC_1] = cellVx[cellC_1] / 2.0;
                    cellVy[cellC_1] = cellVy[cellC_1] / 2.0;

                    // cellArea[cellC_1] = A_min;
                    cellPhi[cellC_1] = phi_eps;
                    cellR[cellC_1] = typeR0[cellType_1];
                    cellFitness[cellC_1][0] = motherFitnessWeight * cellFitness[cellC_1][0] + inherentFitnessWeight * typeFit0[cellType_1];
                    // cellState[cellC_1] = cellState[cellC_1];
                    // original cell

                    divisionTimes[newBornInd].push_back(t);
                    divisionTimes[cellC_1].push_back(t);

                    newBornInd++;
                    newBornCells++;

                    continue;
                } // the end of "if (cellPhi[cellC_1] >= 2.0 *PI)"
                // cell division
                
                // update of cancer cells
                if (cellType_1 == CA_CELL_TYPE) //CA_CELL_TYPE
                {
                    if (cellFitness[cellC_1][0] > Fit_Th_Apop && cellFitnessUpdated[cellC_1][0] > Fit_Th_Apop) // stays alive (fitness simply updates)
                    {
                        cellFitness[cellC_1][0] = cellFitnessUpdated[cellC_1][0];
                        cellState[cellC_1] = CYCLING_STATE;
                    }else if (cellFitness[cellC_1][0] > Fit_Th_Apop && cellFitnessUpdated[cellC_1][0] <= Fit_Th_Apop) // It dies
                    {
                        cellFitness[cellC_1][0] = cellFitnessUpdated[cellC_1][0];
                        cellState[cellC_1] = APOP_STATE;
                    }
                    else // It will saty dead
                    {
                        cellState[cellC_1] = APOP_STATE;
                    }
                    continue;
                }
                // update of cancer cells


                // update of WT cells
                else //WT_CELL_TYPE: fitnesses need to be checked
                {
                    if (cellFitness[cellC_1][0] > Fit_Th_G0) 
                    {
                        if (cellFitnessUpdated[cellC_1][0] > Fit_Th_G0)
                        {
                            cellFitness[cellC_1][0] = cellFitnessUpdated[cellC_1][0];
                            cellState[cellC_1] = CYCLING_STATE;
                        }
                        else // cellFitnessUpdated[cellC_1][0] <= Fit_Th_G0
                        {
                            if (cellPhi[cellC_1] > G1Border * 2.0 *PI) // no go to G0
                            {
                                cellState[cellC_1] = CYCLING_STATE;
                            }
                            else // goes to G0
                            {
                                cellFitness[cellC_1][0] = cellFitnessUpdated[cellC_1][0];
                                cellState[cellC_1] = G0_STATE;
                            }
                        }// the end of "if (cellFitnessUpdated[cellC_1][0] > Fit_Th_G1_arr){}"
                        continue;
                    } // the end of "if (cellFitness[cellC_1][0] > Fit_Th_G1_arr){}"

                    if (cellFitness[cellC_1][0] <= Fit_Th_G0 &&
                        cellFitness[cellC_1][0] >  Fit_Th_Apop) // if it is in G0
                    {
                        if (cellFitnessUpdated[cellC_1][0] <= Fit_Th_G0 &&
                            cellFitnessUpdated[cellC_1][0] >  Fit_Th_Apop) // stays in G0
                        {
                            cellFitness[cellC_1][0] = cellFitnessUpdated[cellC_1][0];
                            cellState[cellC_1] = G0_STATE;
                        }
                        else if (cellFitnessUpdated[cellC_1][0] >  Fit_Th_G0) // stays in G0; cannot go higher
                        {
                            cellState[cellC_1] = G0_STATE;
                        }
                        else if (cellFitnessUpdated[cellC_1][0] < Fit_Th_Apop) // from G0 goes to Apop
                        {
                            cellFitness[cellC_1][0] = cellFitnessUpdated[cellC_1][0];
                            cellState[cellC_1] = APOP_STATE;
                        }
                        continue;
                    }
                }
            } // the end of "for (cellC_1 = 0; cellC_1 < NCells; cellC_1++)" for State update and cell division operations.

            NCells += newBornCells;

            if (NCells >= N_UpperLim)
            {
                cout<<"############################\n";
                cout<<"Error! NCells >= N_UpperLim;\n";
                cout<<"############################";
                return 0;
            }
            

            
            ///// Sampling operation //////
            // NO SAMPLING
            ///// Sampling operation //////

            t += dt;
        }
        /////// SIMULATION LOOP /////////////
        // simulation loop comes here


        for (int TypeC = 0; TypeC < NTypes; TypeC++)
        {
            dist_data dist_data_indiv = dist_data_maker(NCells, cellType, cellPhi, cellFitness, cellState, config, TypeC);
            distributions_steps[TypeC].push_back(dist_data_indiv);
        }
        
        if (sampling_step > numbers_for_check)
        {
            equilibration_cond = eq_cond_checker(distributions_steps, numbers_for_check);
        }
        


        if (equilibration_cond == 0)
        {
            sampling_step++;

            sampler_initializer_func(N_UpperLim, NCells, NCells_orig,
                                    cellType, cellX, cellY, 
                                    cellVx, cellVy, 
                                    cellPhi, cellState, cellTheta,
                                    cellFitness, typeFit0, mt_rand);
        }
  
    }

    writeDoubleVectorToFile(cellPhi, NCells_orig, "init_of_init/Phi_init.txt");
    writeIntVectorToFile(cellState, NCells_orig, "init_of_init/State_init.txt");
    writeDoubleVectorToFile(cellTheta, NCells_orig, "init_of_init/Theta_init.txt");
    writeDoubleMatrixToFile(cellFitness, NCells_orig, 2,  "init_of_init/Fit_init.txt");
    
    


    
    // system("python3 dataZipper.py");



    return 0;
}


//////////////////////////////////////////////////////////////////////////
////////////////////////////// FUNCTIONS /////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void sampler_initializer_func(const int N_UpperLim, int NCells, int NCells_orig,
                                vector<int>& cellType, vector<double>& cellX, vector<double>& cellY, 
                                vector<double>& cellVx, vector<double>& cellVy, 
                                vector<double>& cellPhi, vector<int>& cellState, vector<double>& cellTheta,
                                vector<vector<double>>& cellFitness, const vector<double>& typeFit0, std::mt19937 &mt_rand)
{
    // NCellsPerType[0] = 500;
    // NCellsPerType[1] = 50;

    vector<int> NCellsPerType_read;

    readIntVectorFromFile("Init_numbers.csv", NCellsPerType_read);
    int NTypes = NCellsPerType_read.size();

    
    unsigned long random_int;
    double random_float;
    unsigned long MT_MAX = mt_rand.max();
    unsigned long MT_MIN = mt_rand.min();

    double cellArea_val, cellR_val;

    vector<double> A_tot_sq(NTypes);
    vector<double> A_tot(NTypes);
    for (int i = 0; i < NTypes; i++)
    {
        A_tot_sq[i] = 0.0;
        A_tot[i] = 0.0;
    }
    
    vector<double> cellR(NCells);


    int cellC = 0;

    int typeInd = 0;
    double A_min = PI * typeR0[typeInd] * typeR0[typeInd];
    double A_max = PI * typeR2PI[typeInd] * typeR2PI[typeInd];
    while (cellC < NCellsPerType[0])
    {   
        cellType[cellC] = typeInd;

        // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellPhi[cellC] = random_float * (2*PI);

        cellState[cellC] = CYCLING_STATE; // all the cells start from cycling state

        // cellArea[cellC] = A_min + (A_max - A_min) * 0.5 * (1 - cos(cellPhi[cellC]/2.0)); // cosine area independency to phi
        cellArea_val = A_min + (A_max - A_min) * cellPhi[cellC] / (2 * PI); // linear area independency to phi
        cellR_val = pow(cellArea_val / PI, 0.5);
        cellR[cellC] = cellR_val;
        A_tot_sq[typeInd] += 4 * cellR_val * cellR_val;
        A_tot[typeInd] += PI * cellR_val * cellR_val;

        // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellTheta[cellC] = random_float * (2*PI);

        cellVx[cellC] = 0.0;
        cellVy[cellC] = 0.0;

        cellFitness[cellC][0] = typeFit0[typeInd];
        cellFitness[cellC][1] = 0.0;

        cellC++;
    }

    typeInd = 1;
    A_min = PI * typeR0[typeInd] * typeR0[typeInd];
    A_max = PI * typeR2PI[typeInd] * typeR2PI[typeInd];
    while (cellC < NCellsPerType[0] + NCellsPerType[1])
    {   
        cellType[cellC] = typeInd;

        // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellPhi[cellC] = random_float * (2*PI);

        cellState[cellC] = CYCLING_STATE; // all the cells start from cycling state

        // cellArea[cellC] = A_min + (A_max - A_min) * 0.5 * (1 - cos(cellPhi[cellC]/2.0));  // cosine area independency to phi
        cellArea_val = A_min + (A_max - A_min) * cellPhi[cellC] / (2 * PI); // linear area independency to phi
        cellR_val = pow(cellArea_val / PI, 0.5);
        cellR[cellC] = cellR_val;
        A_tot_sq[typeInd] += 4 * cellR_val * cellR_val;
        A_tot[typeInd] += PI * cellR_val * cellR_val;

        // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellTheta[cellC] = random_float * (2*PI);

        cellVx[cellC] = 0.0;
        cellVy[cellC] = 0.0;

        cellFitness[cellC][0] = typeFit0[typeInd];
        cellFitness[cellC][1] = 0.0;

        cellC++;
    }

    while (cellC < N_UpperLim)
    {
        cellPhi[cellC] = 0;
        cellState[cellC] = 0;

        cellType[cellC] = NULL_CELL_TYPE;

        // cellArea[cellC] = 0;
        // cellR[cellC] = 0;
        cellTheta[cellC] = 0;

        cellX[cellC] = 0;
        cellY[cellC] = 0;

        cellVx[cellC] = 0;
        cellVy[cellC] = 0;

        cellFitness[cellC][0] = 0;
        cellFitness[cellC][1] = 0;

        cellC++;
    }

    ////// initialization of X and Y //////
    // double Lx , Ly;
    double increase_coef = 1.0;
    double R_tot;
    // R_tot = pow( (A_tot_sq[0] + A_tot_sq[1]) / PI, 0.5);
    R_tot = pow( increase_coef*(A_tot[0] + A_tot[1]) / PI, 0.5);
    // Lx = 2.0 * R_tot;
    // Ly = 2.0 * R_tot;

    // finding h (border of WT and Cancer cells)
    double h = R_tot;
    double dh = 0.001 * R_tot;
    double a = 0.0;
    
    // while (a < A_tot_sq[1])
    while (a < A_tot[1]* increase_coef)
    {
        a = R_tot * R_tot * acos(h/R_tot) - h * pow(R_tot * R_tot - h * h , 0.5);
        h = h - dh;
    }
    // finding h (border of WT and Cancer cells)

    double overlap;

    cellC = 0;
    while (cellC < NCellsPerType[0] + NCellsPerType[1])
    {   
        typeInd = cellType[cellC];
        
        int repeat_cond, out_cond, not_part_cond, too_close_cond;
        double x, y;

        do
        {
            // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            x = -R_tot + (2 * R_tot) * random_float;

            // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            y = -R_tot + (2 * R_tot) * random_float;

            out_cond =  ( (x*x + y*y) > (R_tot*R_tot) );
            if (typeInd == 0)
            {
                not_part_cond = (y > h);
            }
            else if (typeInd == 1)
            {
                not_part_cond = (y <= h);
            }

            too_close_cond = 0;
            for (int j = 0; j < cellC; j++)
            {
                if ((x-cellX[j])*(x-cellX[j]) + (y-cellY[j])*(y-cellY[j]) < pow(0.7 * (cellR[cellC]+cellR[j]), 2)  )
                {
                    too_close_cond = 1;
                    break;
                }
                
            }
            
            repeat_cond = out_cond || not_part_cond || too_close_cond;

        } while (repeat_cond);
        
        
        cellX[cellC] = x;
        cellY[cellC] = y;

        cellC++;
    }
    ////// initialization of X and Y //////

    //// This block is for windows:
    // mkdir(ppDataFolderName.c_str()); //making data folder
    //// This block is for Linux:
    mkdir("init", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making backup_resume folder

    writeIntVectorToFile(cellType, NCells, "init/Type_init.txt");
    writeDoubleVectorToFile(cellX, NCells, "init/X_init.txt");
    writeDoubleVectorToFile(cellY, NCells, "init/Y_init.txt");
    writeDoubleVectorToFile(cellVx, NCells, "init/Vx_init.txt");
    writeDoubleVectorToFile(cellVy, NCells, "init/Vy_init.txt");
    writeDoubleVectorToFile(cellPhi, NCells, "init/Phi_init.txt");
    writeIntVectorToFile(cellState, NCells, "init/State_init.txt");
    // writeDoubleVectorToFile(cellR, NCells, "init/R_init.txt");
    writeDoubleVectorToFile(cellTheta, NCells, "init/Theta_init.txt");
    writeDoubleMatrixToFile(cellFitness, NCells, 2,  "init/Fit_init.txt");
}

int eq_cond_checker(const vector<vector<dist_data>> distributions_steps, const int numbers_for_check)
{
    int condition = 1;

    double alpha = 0.05;
    // double c_alpha = sqrt(-0.5 * log(alpha/2.0));

    double threshold_val;
    double maxDiff_eps = 1e-7;

    int NTypes = distributions_steps.size();
    int NSamples = distributions_steps[0].size();

    vector<vector<int>> cond_mat(numbers_for_check, vector<int>(numbers_for_check));

    vector<double> data_sample_1;
    vector<double> data_sample_2;

    double maxDiff;
    
    // loop on types (WT and C)
    for (int typeC = 0; typeC < NTypes; typeC++)
    {
        
        //// phi_tot ////
        // matrices initialization
        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                cond_mat[rowC][colC] = 1;
            }
        }
        data_sample_1.clear();
        data_sample_2.clear();
        // matrices initialization

        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            data_sample_1.clear();
            data_sample_1 = distributions_steps[typeC][NSamples - rowC].phi_tot;
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                data_sample_2.clear();
                data_sample_2 = distributions_steps[typeC][NSamples - colC].phi_tot;
                maxDiff = ksTest(data_sample_1, data_sample_2);
                if (abs(maxDiff)<maxDiff_eps)
                {
                    cond_mat[rowC][colC] = 1;
                }
                else
                {
                    threshold_val = sqrt(-((data_sample_1.size()+data_sample_2.size())/(2.0 * data_sample_1.size()*data_sample_2.size())) * log(alpha/2.0));
                    if (maxDiff > threshold_val)
                    {
                        cond_mat[rowC][colC] = 0;
                    } else
                    {
                        cond_mat[rowC][colC] = 1;
                    }
                }
                condition = condition * cond_mat[rowC][colC];
            }
        }
        //// phi_tot ////

        //// fit_tot ////
        // matrices initialization
        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                cond_mat[rowC][colC] = 1;
            }
        }
        data_sample_1.clear();
        data_sample_2.clear();
        // matrices initialization

        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            data_sample_1.clear();
            data_sample_1 = distributions_steps[typeC][NSamples - rowC].fit_tot;
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                data_sample_2.clear();
                data_sample_2 = distributions_steps[typeC][NSamples - colC].fit_tot;
                maxDiff = ksTest(data_sample_1, data_sample_2);
                if (abs(maxDiff)<maxDiff_eps)
                {
                    cond_mat[rowC][colC] = 1;
                }
                else
                {
                    threshold_val = sqrt(-((data_sample_1.size()+data_sample_2.size())/(2.0 * data_sample_1.size()*data_sample_2.size())) * log(alpha/2.0));
                    if (maxDiff > threshold_val)
                    {
                        cond_mat[rowC][colC] = 0;
                    } else
                    {
                        cond_mat[rowC][colC] = 1;
                    }
                }
                condition = condition * cond_mat[rowC][colC];
            }
        }
        //// fit_tot ////

        //// phi_G0 ////
        // matrices initialization
        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                cond_mat[rowC][colC] = 1;
            }
        }
        data_sample_1.clear();
        data_sample_2.clear();
        // matrices initialization

        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            data_sample_1.clear();
            data_sample_1 = distributions_steps[typeC][NSamples - rowC].phi_G0;
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                data_sample_2.clear();
                data_sample_2 = distributions_steps[typeC][NSamples - colC].phi_G0;
                maxDiff = ksTest(data_sample_1, data_sample_2);
                if (abs(maxDiff)<maxDiff_eps)
                {
                    cond_mat[rowC][colC] = 1;
                }
                else
                {
                    threshold_val = sqrt(-((data_sample_1.size()+data_sample_2.size())/(2.0 * data_sample_1.size()*data_sample_2.size())) * log(alpha/2.0));
                    if (maxDiff > threshold_val)
                    {
                        cond_mat[rowC][colC] = 0;
                    } else
                    {
                        cond_mat[rowC][colC] = 1;
                    }
                }
                condition = condition * cond_mat[rowC][colC];
            }
        }
        //// phi_G0 ////

        //// fit_G0 ////
        // matrices initialization
        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                cond_mat[rowC][colC] = 1;
            }
        }
        data_sample_1.clear();
        data_sample_2.clear();
        // matrices initialization

        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            data_sample_1.clear();
            data_sample_1 = distributions_steps[typeC][NSamples - rowC].fit_G0;
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                data_sample_2.clear();
                data_sample_2 = distributions_steps[typeC][NSamples - colC].fit_G0;
                maxDiff = ksTest(data_sample_1, data_sample_2);
                if (abs(maxDiff)<maxDiff_eps)
                {
                    cond_mat[rowC][colC] = 1;
                }
                else
                {
                    threshold_val = sqrt(-((data_sample_1.size()+data_sample_2.size())/(2.0 * data_sample_1.size()*data_sample_2.size())) * log(alpha/2.0));
                    if (maxDiff > threshold_val)
                    {
                        cond_mat[rowC][colC] = 0;
                    } else
                    {
                        cond_mat[rowC][colC] = 1;
                    }
                }
                condition = condition * cond_mat[rowC][colC];
            }
        }
        //// fit_G0 ////

        //// phi_G1 ////
        // matrices initialization
        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                cond_mat[rowC][colC] = 1;
            }
        }
        data_sample_1.clear();
        data_sample_2.clear();
        // matrices initialization

        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            data_sample_1.clear();
            data_sample_1 = distributions_steps[typeC][NSamples - rowC].phi_G1;
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                data_sample_2.clear();
                data_sample_2 = distributions_steps[typeC][NSamples - colC].phi_G1;
                maxDiff = ksTest(data_sample_1, data_sample_2);
                if (abs(maxDiff)<maxDiff_eps)
                {
                    cond_mat[rowC][colC] = 1;
                }
                else
                {
                    threshold_val = sqrt(-((data_sample_1.size()+data_sample_2.size())/(2.0 * data_sample_1.size()*data_sample_2.size())) * log(alpha/2.0));
                    if (maxDiff > threshold_val)
                    {
                        cond_mat[rowC][colC] = 0;
                    } else
                    {
                        cond_mat[rowC][colC] = 1;
                    }
                }
                condition = condition * cond_mat[rowC][colC];
            }
        }
        //// phi_G1 ////

        //// fit_G1 ////
        // matrices initialization
        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                cond_mat[rowC][colC] = 1;
            }
        }
        data_sample_1.clear();
        data_sample_2.clear();
        // matrices initialization

        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            data_sample_1.clear();
            data_sample_1 = distributions_steps[typeC][NSamples - rowC].fit_G1;
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                data_sample_2.clear();
                data_sample_2 = distributions_steps[typeC][NSamples - colC].fit_G1;
                maxDiff = ksTest(data_sample_1, data_sample_2);
                if (abs(maxDiff)<maxDiff_eps)
                {
                    cond_mat[rowC][colC] = 1;
                }
                else
                {
                    threshold_val = sqrt(-((data_sample_1.size()+data_sample_2.size())/(2.0 * data_sample_1.size()*data_sample_2.size())) * log(alpha/2.0));
                    if (maxDiff > threshold_val)
                    {
                        cond_mat[rowC][colC] = 0;
                    } else
                    {
                        cond_mat[rowC][colC] = 1;
                    }
                }
                condition = condition * cond_mat[rowC][colC];
            }
        }
        //// fit_G1 ////

        //// phi_G2 ////
        // matrices initialization
        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                cond_mat[rowC][colC] = 1;
            }
        }
        data_sample_1.clear();
        data_sample_2.clear();
        // matrices initialization

        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            data_sample_1.clear();
            data_sample_1 = distributions_steps[typeC][NSamples - rowC].phi_G2;
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                data_sample_2.clear();
                data_sample_2 = distributions_steps[typeC][NSamples - colC].phi_G2;
                maxDiff = ksTest(data_sample_1, data_sample_2);
                if (abs(maxDiff)<maxDiff_eps)
                {
                    cond_mat[rowC][colC] = 1;
                }
                else
                {
                    threshold_val = sqrt(-((data_sample_1.size()+data_sample_2.size())/(2.0 * data_sample_1.size()*data_sample_2.size())) * log(alpha/2.0));
                    if (maxDiff > threshold_val)
                    {
                        cond_mat[rowC][colC] = 0;
                    } else
                    {
                        cond_mat[rowC][colC] = 1;
                    }
                }
                condition = condition * cond_mat[rowC][colC];
            }
        }
        //// phi_G2 ////

        //// fit_G2 ////
        // matrices initialization
        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                cond_mat[rowC][colC] = 1;
            }
        }
        data_sample_1.clear();
        data_sample_2.clear();
        // matrices initialization

        for (int rowC = 0; rowC < numbers_for_check; rowC++)
        {
            data_sample_1.clear();
            data_sample_1 = distributions_steps[typeC][NSamples - rowC].fit_G2;
            for (int colC = 0; colC < numbers_for_check; colC++)
            {
                data_sample_2.clear();
                data_sample_2 = distributions_steps[typeC][NSamples - colC].fit_G2;
                maxDiff = ksTest(data_sample_1, data_sample_2);
                if (abs(maxDiff)<maxDiff_eps)
                {
                    cond_mat[rowC][colC] = 1;
                }
                else
                {
                    threshold_val = sqrt(-((data_sample_1.size()+data_sample_2.size())/(2.0 * data_sample_1.size()*data_sample_2.size())) * log(alpha/2.0));
                    if (maxDiff > threshold_val)
                    {
                        cond_mat[rowC][colC] = 0;
                    } else
                    {
                        cond_mat[rowC][colC] = 1;
                    }
                }
                condition = condition * cond_mat[rowC][colC];
            }
        }
        //// fit_G2 ////

    }

    return condition;
}

// Function to calculate the empirical distribution function (EDF)
std::vector<std::pair<double, double>> calculateEDF(const std::vector<double>& sample) {
    std::vector<std::pair<double, double>> edf;
    size_t n = sample.size();
    for (size_t i = 0; i < n; ++i) {
        edf.emplace_back(sample[i], static_cast<double>(i + 1) / n);
    }
    return edf;
}

// Function to perform the two-sample Kolmogorov-Smirnov test
double ksTest(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    // Sort both samples
    std::vector<double> sortedSample1 = sample1;
    std::vector<double> sortedSample2 = sample2;
    std::sort(sortedSample1.begin(), sortedSample1.end());
    std::sort(sortedSample2.begin(), sortedSample2.end());

    // Calculate EDFs
    auto edf1 = calculateEDF(sortedSample1);
    auto edf2 = calculateEDF(sortedSample2);

    // Merge and find the maximum difference between EDFs
    size_t i = 0, j = 0;
    double maxDiff = 0.0;

    while (i < edf1.size() && j < edf2.size()) {
        if (edf1[i].first < edf2[j].first) {
            maxDiff = std::max(maxDiff, std::abs(edf1[i].second - (j > 0 ? edf2[j - 1].second : 0.0)));
            ++i;
        } else if (edf1[i].first > edf2[j].first) {
            maxDiff = std::max(maxDiff, std::abs(edf2[j].second - (i > 0 ? edf1[i - 1].second : 0.0)));
            ++j;
        } else {
            maxDiff = std::max(maxDiff, std::abs(edf1[i].second - edf2[j].second));
            ++i;
            ++j;
        }
    }

    // Check remaining EDFs if necessary
    while (i < edf1.size()) {
        maxDiff = std::max(maxDiff, std::abs(edf1[i].second - edf2.back().second));
        ++i;
    }
    while (j < edf2.size()) {
        maxDiff = std::max(maxDiff, std::abs(edf2[j].second - edf1.back().second));
        ++j;
    }

    return maxDiff;
}


dist_data dist_data_maker(const int NCells, const vector<int>& cellType,  const vector<double>& cellPhi, const vector<vector<double>>& cellFitness, \
                          const vector<int>& cellState, const Config config, const int Type)
{
    dist_data dist_data_var;

    int is_G0, is_G1, is_G2;

    for (int cellC_1 = 0; cellC_1 < NCells; cellC_1++)
    {
        int cellType_1 = cellType[cellC_1];

        if (cellType_1 == Type) // Only for the type Type
        {

            if (cellType_1 == WT_CELL_TYPE)
            {
                is_G0 = (cellPhi[cellC_1] < config.G1Border*2*PI) && (cellFitness[cellC_1][0] < config.Fit_Th_G0) && (cellState[cellC_1] > APOP_STATE);
                is_G1 = (cellPhi[cellC_1] < config.G1Border*2*PI) && (cellFitness[cellC_1][0] > config.Fit_Th_G0) && (cellState[cellC_1] > APOP_STATE);
                is_G2 = (cellPhi[cellC_1] > config.G1Border*2*PI) && (cellState[cellC_1] > APOP_STATE);
            }
            else if (cellType_1 == CA_CELL_TYPE)
            {
                is_G0 = 0;
                is_G1 = (cellPhi[cellC_1] < config.G1Border*2*PI) && (cellState[cellC_1] > APOP_STATE);
                is_G2 = (cellPhi[cellC_1] > config.G1Border*2*PI) && (cellState[cellC_1] > APOP_STATE);
            }

            if (is_G0)
            {
                dist_data_var.phi_G0.push_back(cellPhi[cellC_1]);
                dist_data_var.fit_G0.push_back(cellFitness[cellC_1][0]);
            } else
            if (is_G1)
            {
                dist_data_var.phi_G1.push_back(cellPhi[cellC_1]);
                dist_data_var.fit_G1.push_back(cellFitness[cellC_1][0]);
            } else
            if (is_G2)
            {
                dist_data_var.phi_G2.push_back(cellPhi[cellC_1]);
                dist_data_var.fit_G2.push_back(cellFitness[cellC_1][0]);
            }

        }

    }

    return dist_data_var;
}

int vector_condition_checker(const vector<double> VEC, const int numbers_for_check)
{
    int condition;
    double excluded_mean, excluded_SEM, excluded_std, excluded_val;
    int VEC_len = VEC.size();

    if (VEC_len < numbers_for_check)
    {
        condition = 0;
    }
    else
    {
        condition = 1;

        for (int i = 0; i < numbers_for_check; i++)
        {
            excluded_val = VEC[VEC_len - i];
            excluded_mean = 0.0;
            for (int j = 0; j < numbers_for_check; j++)
            {
                if (j != i)
                {
                    excluded_mean = excluded_mean + VEC[VEC_len - j];
                }
            }
            excluded_mean = excluded_mean / (numbers_for_check-1);
            excluded_std = 0.0;
            for (int j = 0; j < numbers_for_check; j++)
            {
                if (j != i)
                {
                    excluded_std = excluded_std + pow((VEC[VEC_len - j]-excluded_mean), 2);
                }
            }
            excluded_std = sqrt(excluded_std/(numbers_for_check-2.0));
            excluded_SEM = excluded_std / sqrt(numbers_for_check-2.0);

            condition = condition * (int)((excluded_val>excluded_mean-excluded_SEM) && (excluded_val<excluded_mean+excluded_SEM));
        }
    }
    return condition;
}

void simulationDataReader(int* NSitesPtr, double* LxPtr, double* LyPtr, double* AlphaPtr, double* KbPtr, double* TemPtr,  int* NumCellsPtr, \
                          double* AvgCellAreaPtr, double* LambdaPtr, long* maxMCStepsPtr, int* samplesPerWritePtr, \
                          int* printingTimeIntervalPtr, int* numLinkedListPtr, string* initConfigPtr)
{   
    // int L_read;
    // int NumCells_read;
    // int samplesPerWrite_read;

    fstream newfile;
    newfile.open("simulationData_vec.csv",ios::in); //open a file to perform read operation using file object
    if (newfile.is_open()){ //checking whether the file is open
        string tp;
        while(getline(newfile, tp)){ //read data from file object and put it into string.
            // cout << tp << "\n"; //print the data of the string
            // if (strstr((const char)tp, "NSites = "))
            const char* tpChar = tp.c_str();
            if (strstr(tpChar, "NSites = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *NSitesPtr = std::stoi(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "Lx = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *LxPtr = std::stod(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "Ly = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *LyPtr = std::stod(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "Alpha = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *AlphaPtr = std::stod(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "Kb = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *KbPtr = std::stod(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "Tem = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *TemPtr = std::stod(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "NumCells = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *NumCellsPtr = std::stoi(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "AvgCellArea = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *AvgCellAreaPtr = std::stod(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "Lambda = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *LambdaPtr = std::stod(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "SweepLength = "))
            {
                // Do nothing
                continue;
            }
            if (strstr(tpChar, "maxMCSteps = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *maxMCStepsPtr = std::stoi(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "samplesPerWrite = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *samplesPerWritePtr = std::stoi(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "printingTimeInterval = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *printingTimeIntervalPtr = std::stoi(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "numLinkedList = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *numLinkedListPtr = std::stoi(num_str);  // convert substring to int
                continue;
            }
            if (strstr(tpChar, "initConfig = "))
            {
                std::size_t pos = tp.find('=');  // find position of '='
                std::string num_str = tp.substr(pos+2);  // extract substring starting from position after '='
                *initConfigPtr = num_str;
                continue;
            }
        }
    }
    newfile.close(); //close the file object.

    // *L_read_ptr = L_read;
    // *NumCells_read_ptr = NumCells_read;
    // *samplesPerWrite_read_ptr = samplesPerWrite_read;
}

std::vector<double> parseVector(const std::string& str) {
    std::vector<double> result;
    std::stringstream ss(str.substr(1, str.size() - 2)); // Remove the square brackets
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stod(item));
    }
    return result;
}

std::vector<std::vector<double>> parseMatrix(const std::string& str) {
    std::vector<std::vector<double>> result;
    std::stringstream ss(str.substr(1, str.size() - 2)); // Remove the square brackets
    std::string row;
    while (std::getline(ss, row, ';')) {
        result.push_back(parseVector(row));
    }
    return result;
}

void trim(std::string& str) {
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), str.end());
}

void readConfigFile(const std::string& filename, Config& config) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key;
        std::getline(ss, key, '=');
        std::string value;
        std::getline(ss, value);

        trim(key);
        trim(value);

        if (key == "N_UpperLim") {
            config.N_UpperLim = std::stoi(value);
        } else if (key == "NTypes") {
            config.NTypes = std::stoi(value);
        } else if (key == "typeR0") {
            config.typeR0 = parseVector(value);
        } else if (key == "typeR2PI") {
            config.typeR2PI = parseVector(value);
        } else if (key == "typeTypeEpsilon") {
            config.typeTypeEpsilon = parseMatrix(value);
        //////////////////////////////////////////////////
        } else if (key == "typeGamma") {
            config.typeGamma = parseVector(value);
        } else if (key == "typeTypeGammaCC") {
            config.typeTypeGammaCC = parseMatrix(value);
        } else if (key == "typeTypeF_rep_max") {
            config.typeTypeF_rep_max = parseMatrix(value);
        } else if (key == "typeTypeF_abs_max") {
            config.typeTypeF_abs_max = parseMatrix(value);
        } else if (key == "R_eq_coef") {
            config.R_eq_coef = std::stod(value);
        } else if (key == "R_cut_coef_force") {
            config.R_cut_coef_force = std::stod(value);
        //////////////////////////////////////////////////
        } else if (key == "typeFm") {
            config.typeFm = parseVector(value);
        } else if (key == "typeDr") {
            config.typeDr = parseVector(value);
        //////////////////////////////////////////////////
        } else if (key == "G1Border") {
            config.G1Border = stod(value);
        //////////////////////////////////////////////////
        } else if (key == "typeOmega") {
            config.typeOmega = parseVector(value);
        } else if (key == "typeBarrierW") {
            config.typeBarrierW = parseVector(value);
        } else if (key == "typeSigmaPhi") {
            config.typeSigmaPhi = parseVector(value);
        } else if (key == "typeSigmaFit") {
            config.typeSigmaFit = parseVector(value);
        } else if (key == "barrierPeakCoef") {
            config.barrierPeakCoef = stod(value);

        } else if (key == "typeFit0") {
            config.typeFit0 = parseVector(value);
        } else if (key == "Fit_Th_Wall") {
            config.Fit_Th_Wall = std::stod(value);
        } else if (key == "Fit_Th_G0") {
            config.Fit_Th_G0 = std::stod(value);
        } else if (key == "Fit_Th_Diff") {
            config.Fit_Th_Diff = std::stod(value);
        } else if (key == "Fit_Th_Apop") {
            config.Fit_Th_Apop = std::stod(value);
        //////////////////////////////////////////////////
        } else if (key == "maxTime") {
            config.maxTime = std::stod(value);
        } else if (key == "dt") {
            config.dt = std::stod(value);
        } else if (key == "dt_sample") {
            config.dt_sample = std::stod(value);
        } else if (key == "samplesPerWrite") {
            config.samplesPerWrite = std::stoi(value);
        } else if (key == "writePerZip") {
            config.writePerZip = std::stoi(value);
        } else if (key == "printingTimeInterval") {
            config.printingTimeInterval = std::stod(value);
        //////////////////////////////////////////////////
        } else if (key == "R_cut_coef_game") {
            config.R_cut_coef_game = std::stod(value);
        } else if (key == "tau") {
            config.tau = std::stod(value);
        } else if (key == "typeTypePayOff_mat_real_C") {
            config.typeTypePayOff_mat_real_C = parseMatrix(value);
        } else if (key == "typeTypePayOff_mat_real_F1") {
            config.typeTypePayOff_mat_real_F1 = parseMatrix(value);
        } else if (key == "typeTypePayOff_mat_real_F2") {
            config.typeTypePayOff_mat_real_F2 = parseMatrix(value);
        } else if (key == "typeTypePayOff_mat_imag_C") {
            config.typeTypePayOff_mat_imag_C = parseMatrix(value);
        } else if (key == "typeTypePayOff_mat_imag_F1") {
            config.typeTypePayOff_mat_imag_F1 = parseMatrix(value);
        } else if (key == "typeTypePayOff_mat_imag_F2") {
            config.typeTypePayOff_mat_imag_F2 = parseMatrix(value);
        //////////////////////////////////////////////////
        } else if (key == "newBornFitKey") {
            config.newBornFitKey = value;
        //////////////////////////////////////////////////
        } else if (key == "initConfig") {
            config.initConfig = value;
        
        }
    }
}

void initializer(const int N_UpperLim, int* NCellsPtr, vector<int>& NCellsPerType,
                 const vector<double> typeR0, const vector<double> typeR2PI, 
                 vector<int>& cellType, vector<double>& cellX, vector<double>& cellY, 
                 vector<double>& cellVx, vector<double>& cellVy, 
                 vector<double>& cellPhi, vector<int>& cellState, vector<double>& cellTheta,
                 vector<vector<double>>& cellFitness, const vector<double>& typeFit0, std::mt19937 &mt_rand)
{
    // NCellsPerType[0] = 500;
    // NCellsPerType[1] = 50;

    vector<int> NCellsPerType_read;

    readIntVectorFromFile("Init_numbers.csv", NCellsPerType_read);
    int NTypes = NCellsPerType_read.size();

    for (int i = 0; i < NTypes; i++)
    {
        NCellsPerType[i] = NCellsPerType_read[i];
    }

    int NCells = NCellsPerType[0] + NCellsPerType[1];
    
    *NCellsPtr = NCells;

    // std::srand(static_cast<unsigned int>(std::time(0)));
    
    unsigned long random_int;
    double random_float;
    unsigned long MT_MAX = mt_rand.max();
    unsigned long MT_MIN = mt_rand.min();

    double cellArea_val, cellR_val;

    vector<double> A_tot_sq(NTypes);
    vector<double> A_tot(NTypes);
    for (int i = 0; i < NTypes; i++)
    {
        A_tot_sq[i] = 0.0;
        A_tot[i] = 0.0;
    }
    
    vector<double> cellR(NCells);


    int cellC = 0;

    int typeInd = 0;
    double A_min = PI * typeR0[typeInd] * typeR0[typeInd];
    double A_max = PI * typeR2PI[typeInd] * typeR2PI[typeInd];
    while (cellC < NCellsPerType[0])
    {   
        cellType[cellC] = typeInd;

        // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellPhi[cellC] = random_float * (2*PI);

        cellState[cellC] = CYCLING_STATE; // all the cells start from cycling state

        // cellArea[cellC] = A_min + (A_max - A_min) * 0.5 * (1 - cos(cellPhi[cellC]/2.0)); // cosine area independency to phi
        cellArea_val = A_min + (A_max - A_min) * cellPhi[cellC] / (2 * PI); // linear area independency to phi
        cellR_val = pow(cellArea_val / PI, 0.5);
        cellR[cellC] = cellR_val;
        A_tot_sq[typeInd] += 4 * cellR_val * cellR_val;
        A_tot[typeInd] += PI * cellR_val * cellR_val;

        // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellTheta[cellC] = random_float * (2*PI);

        cellVx[cellC] = 0.0;
        cellVy[cellC] = 0.0;

        cellFitness[cellC][0] = typeFit0[typeInd];
        cellFitness[cellC][1] = 0.0;

        cellC++;
    }

    typeInd = 1;
    A_min = PI * typeR0[typeInd] * typeR0[typeInd];
    A_max = PI * typeR2PI[typeInd] * typeR2PI[typeInd];
    while (cellC < NCellsPerType[0] + NCellsPerType[1])
    {   
        cellType[cellC] = typeInd;

        // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellPhi[cellC] = random_float * (2*PI);

        cellState[cellC] = CYCLING_STATE; // all the cells start from cycling state

        // cellArea[cellC] = A_min + (A_max - A_min) * 0.5 * (1 - cos(cellPhi[cellC]/2.0));  // cosine area independency to phi
        cellArea_val = A_min + (A_max - A_min) * cellPhi[cellC] / (2 * PI); // linear area independency to phi
        cellR_val = pow(cellArea_val / PI, 0.5);
        cellR[cellC] = cellR_val;
        A_tot_sq[typeInd] += 4 * cellR_val * cellR_val;
        A_tot[typeInd] += PI * cellR_val * cellR_val;

        // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellTheta[cellC] = random_float * (2*PI);

        cellVx[cellC] = 0.0;
        cellVy[cellC] = 0.0;

        cellFitness[cellC][0] = typeFit0[typeInd];
        cellFitness[cellC][1] = 0.0;

        cellC++;
    }

    while (cellC < N_UpperLim)
    {
        cellPhi[cellC] = 0;
        cellState[cellC] = 0;

        cellType[cellC] = NULL_CELL_TYPE;

        // cellArea[cellC] = 0;
        // cellR[cellC] = 0;
        cellTheta[cellC] = 0;

        cellX[cellC] = 0;
        cellY[cellC] = 0;

        cellVx[cellC] = 0;
        cellVy[cellC] = 0;

        cellFitness[cellC][0] = 0;
        cellFitness[cellC][1] = 0;

        cellC++;
    }

    ////// initialization of X and Y //////
    // double Lx , Ly;
    double increase_coef = 1.0;
    double R_tot;
    // R_tot = pow( (A_tot_sq[0] + A_tot_sq[1]) / PI, 0.5);
    R_tot = pow( increase_coef*(A_tot[0] + A_tot[1]) / PI, 0.5);
    // Lx = 2.0 * R_tot;
    // Ly = 2.0 * R_tot;

    // finding h (border of WT and Cancer cells)
    double h = R_tot;
    double dh = 0.001 * R_tot;
    double a = 0.0;
    
    // while (a < A_tot_sq[1])
    while (a < A_tot[1]* increase_coef)
    {
        a = R_tot * R_tot * acos(h/R_tot) - h * pow(R_tot * R_tot - h * h , 0.5);
        h = h - dh;
    }
    // finding h (border of WT and Cancer cells)

    double overlap;

    cellC = 0;
    while (cellC < NCellsPerType[0] + NCellsPerType[1])
    {   
        typeInd = cellType[cellC];
        
        int repeat_cond, out_cond, not_part_cond, too_close_cond;
        double x, y;

        do
        {
            // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            x = -R_tot + (2 * R_tot) * random_float;

            // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            y = -R_tot + (2 * R_tot) * random_float;

            out_cond =  ( (x*x + y*y) > (R_tot*R_tot) );
            if (typeInd == 0)
            {
                not_part_cond = (y > h);
            }
            else if (typeInd == 1)
            {
                not_part_cond = (y <= h);
            }

            too_close_cond = 0;
            for (int j = 0; j < cellC; j++)
            {
                if ((x-cellX[j])*(x-cellX[j]) + (y-cellY[j])*(y-cellY[j]) < pow(0.7 * (cellR[cellC]+cellR[j]), 2)  )
                {
                    too_close_cond = 1;
                    break;
                }
                
            }
            
            repeat_cond = out_cond || not_part_cond || too_close_cond;

        } while (repeat_cond);
        
        
        cellX[cellC] = x;
        cellY[cellC] = y;

        cellC++;
    }
    ////// initialization of X and Y //////

    //// This block is for windows:
    // mkdir(ppDataFolderName.c_str()); //making data folder
    //// This block is for Linux:
    mkdir("init", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making backup_resume folder

    writeIntVectorToFile(cellType, NCells, "init/Type_init.txt");
    writeDoubleVectorToFile(cellX, NCells, "init/X_init.txt");
    writeDoubleVectorToFile(cellY, NCells, "init/Y_init.txt");
    writeDoubleVectorToFile(cellVx, NCells, "init/Vx_init.txt");
    writeDoubleVectorToFile(cellVy, NCells, "init/Vy_init.txt");
    writeDoubleVectorToFile(cellPhi, NCells, "init/Phi_init.txt");
    writeIntVectorToFile(cellState, NCells, "init/State_init.txt");
    // writeDoubleVectorToFile(cellR, NCells, "init/R_init.txt");
    writeDoubleVectorToFile(cellTheta, NCells, "init/Theta_init.txt");
    writeDoubleMatrixToFile(cellFitness, NCells, 2,  "init/Fit_init.txt");
}

void initial_read(const int N_UpperLim, int* NCellsPtr, vector<int>& NCellsPerType,
                 const vector<double> typeR0, const vector<double> typeR2PI, 
                 vector<int>& cellType, vector<double>& cellX, vector<double>& cellY, 
                 vector<double>& cellVx, vector<double>& cellVy, 
                 vector<double>& cellPhi, vector<int>& cellState, vector<double>& cellTheta,
                 vector<vector<double>>& cellFitness)
{
    
    vector<int> cellType_read;
    vector<double> cellX_read;
    vector<double> cellY_read;
    vector<double> cellVx_read;
    vector<double> cellVy_read;
    vector<double> cellPhi_read;
    vector<int> cellState_read;
    vector<double> cellTheta_read;
    // vector<double> cellR_read;
    vector<vector<double>> cellFitness_read;

    readIntVectorFromFile("init/Type_init.txt", cellType_read);
    readDoubleVectorFromFile("init/X_init.txt", cellX_read);
    readDoubleVectorFromFile("init/Y_init.txt", cellY_read);
    // readDoubleVectorFromFile("init/R_init.txt", cellR_read);
    readDoubleVectorFromFile("init/Vx_init.txt", cellVx_read);
    readDoubleVectorFromFile("init/Vy_init.txt", cellVy_read);
    readDoubleVectorFromFile("init/Phi_init.txt", cellPhi_read);
    readIntVectorFromFile("init/State_init.txt", cellState_read);
    readDoubleVectorFromFile("init/Theta_init.txt", cellTheta_read);
    readDoubleMatrixFromFile("init/Fit_init.txt", cellFitness_read);

    int NCells = cellType_read.size();
    *NCellsPtr = NCells;

    NCellsPerType[0] = 0;
    NCellsPerType[1] = 0;

    for (int cellC = 0; cellC < NCells; cellC++)
    {
        cellType[cellC] = cellType_read[cellC];
        cellX[cellC] = cellX_read[cellC];
        cellY[cellC] = cellY_read[cellC];
        // cellR[cellC] = cellR_read[cellC];
        cellVx[cellC] = cellVx_read[cellC];
        cellVy[cellC] = cellVy_read[cellC];
        cellPhi[cellC] = cellPhi_read[cellC];
        cellState[cellC] = cellState_read[cellC];
        // cellTheta[cellC] = cellTheta_read[cellC];
        cellTheta[cellC] = 0.0;

        cellFitness[cellC][0]  = cellFitness_read[cellC][0];
        cellFitness[cellC][1]  = cellFitness_read[cellC][1];

        NCellsPerType[cellType[cellC]]++;

        // cellArea[cellC] = (PI * cellR[cellC] * cellR[cellC]);
    }

    for (int cellC = NCells; cellC < N_UpperLim; cellC++)
    {
        cellPhi[cellC] = 0;
        cellState[cellC] = 0;

        cellType[cellC] = NULL_CELL_TYPE;

        // cellArea[cellC] = 0;
        // cellR[cellC] = 0;
        cellTheta[cellC] = 0;

        cellX[cellC] = 0;
        cellY[cellC] = 0;

        cellVx[cellC] = 0;
        cellVy[cellC] = 0;

        cellFitness[cellC][0] = 0;
        cellFitness[cellC][1] = 0;
    }


}

void R_Area_calc(const int N_UpperLim, const int NCells, const int NTypes,
                 const vector<double> typeR0, const vector<double> typeR2PI, 
                 const vector<int>& cellType,
                 const vector<double>& cellPhi, const vector<int>& cellState,
                 vector<double>& cellR, vector<double>& cellArea)
{   

    for (int cellC = 0; cellC < N_UpperLim; cellC++) // initial zero-assignment
    {
        cellArea[cellC] = 0.0;
        cellR[cellC] = 0.0;
    }

    vector<double> A_min(NTypes);
    vector<double> A_max(NTypes);
    
    for (int type_C = 0; type_C < NTypes; type_C++)
    {
        A_min[type_C] = PI * typeR0[type_C] * typeR0[type_C];
        A_max[type_C] = PI * typeR2PI[type_C] * typeR2PI[type_C];
    }

    for (int cellC = 0; cellC < NCells; cellC++)
    {
        if (cellState[cellC] == APOP_STATE) // the cell is dead
        {
            cellArea[cellC] = 0.0;
            cellR[cellC] = 0.0;
        }
        else // the cell is alive
        {
            cellArea[cellC] = A_min[cellType[cellC]] + (A_max[cellType[cellC]] - A_min[cellType[cellC]]) * cellPhi[cellC] / (2 * PI); // linear area independency to phi
            cellR[cellC] = pow(cellArea[cellC] / PI, 0.5);
        }
    }
}

void writeIntVectorToFile(const std::vector<int>& vec, int NCells, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int vecSize = vec.size();
    NCells = std::min(NCells, vecSize);

    for (int i = 0; i < NCells; ++i) {
        outFile << vec[i] << std::endl;
    }

    outFile.close();
    // std::cout << "Data written to file: " << filename << std::endl;
}

void writeIntMatrixToFile(const std::vector<std::vector<int>>& mat, const int N_rows_desired, const int N_cols_desired, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int matRows = mat.size();
    int matCols = mat[0].size();
    int N_Rows = std::min(N_rows_desired, matRows);
    int N_Cols = std::min(N_cols_desired, matCols);

    for (int i = 0; i < N_Rows; ++i) {
        for (int j = 0; j < N_Cols; ++j) {
            outFile << mat[i][j];
            if (j < N_Cols - 1) {
                outFile << ", ";
            }
        }
        outFile << std::endl;
    }

    outFile.close();
    // std::cout << "Data written to file: " << filename << std::endl;
}

void writeDoubleVectorToFile(const std::vector<double>& vec, int NCells, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(8);

    int vecSize = vec.size();
    NCells = std::min(NCells, vecSize);

    for (int i = 0; i < NCells; ++i) {
        outFile << vec[i] << std::endl;
    }

    outFile.close();
    // std::cout << "Data written to file: " << filename << std::endl;
}

void writeDoubleMatrixToFile(const std::vector<std::vector<double>>& mat, const int N_rows_desired, const int N_cols_desired, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(8);

    int matRows = mat.size();
    int matCols = mat[0].size();
    int N_Rows = std::min(N_rows_desired, matRows);
    int N_Cols = std::min(N_cols_desired, matCols);

    for (int i = 0; i < N_Rows; ++i) {
        for (int j = 0; j < N_Cols; ++j) {
            outFile << mat[i][j];
            if (j < N_Cols - 1) {
                outFile << ", ";
            }
        }
        outFile << std::endl;
    }

    outFile.close();
    // std::cout << "Data written to file: " << filename << std::endl;
}

void writeFitnessToFile(const std::vector<std::vector<std::vector<double>>>& matrix, const int N_rows_desired, const int N_cols_desired, const std::string& filename) {
    // std::ofstream outFile(filename);

    // if (!outFile.is_open()) {
    //     std::cerr << "Error opening file: " << filename << std::endl;
    //     return;
    // }

    // outFile << std::fixed << std::setprecision(8);

    int bunchLength = matrix.size();
    int N_Rows = N_rows_desired;
    int N_Cols = 2 * bunchLength;

    vector<vector<double>> outlet_matrix(N_Rows, vector<double>(N_Cols));

    int rowC, colC;
    for (int i = 0; i < bunchLength; i++)
    {
        for (int j = 0; j < N_rows_desired; j++)
        {
            rowC = j;
            colC = 2*i;
            outlet_matrix[rowC][colC] = matrix[i][j][0];

            rowC = j;
            colC = 2*i+1;
            outlet_matrix[rowC][colC] = matrix[i][j][1];
        }
        
    }

    writeDoubleMatrixToFile(outlet_matrix, N_Rows, N_Cols, filename);
    

}

void readIntVectorFromFile(const std::string& filename, std::vector<int>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    int value;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (ss >> value) {
            data.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }
    }

    file.close();
}

void readDoubleVectorFromFile(const std::string& filename, std::vector<double>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    double value;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (ss >> value) {
            data.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }
    }

    file.close();
}

void readIntMatrixFromFile(const std::string& filename, std::vector<std::vector<int>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<int> row;
        int value;
        char comma;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        data.push_back(row);
    }

    file.close();
}

void readDoubleMatrixFromFile(const std::string& filename, std::vector<std::vector<double>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        char comma;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        data.push_back(row);
    }

    file.close();
}

std::vector<std::vector<int>> IntTranspose(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return {};

    int rows = matrix.size();
    int cols = matrix[0].size();

    // Create a result vector with transposed dimensions
    std::vector<std::vector<int>> transposed(cols, std::vector<int>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

std::vector<std::vector<double>> DoubleTranspose(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return {};

    int rows = matrix.size();
    int cols = matrix[0].size();

    // Create a result vector with transposed dimensions
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

void dataBunchWriter(const int NCells, \
                     const vector<double> tBunch, \
                     const vector<vector<int>> cellTypeBunch, \
                     const vector<vector<double>> cellXBunch, \
                     const vector<vector<double>> cellYBunch, \
                     const vector<vector<double>> cellVxBunch, \
                     const vector<vector<double>> cellVyBunch, \
                     const vector<vector<double>> cellPhiBunch, \
                     const vector<vector<int>> cellStateBunch, \
                     const vector<vector<vector<double>>> cellFitnessBunch, \
                     const int saved_bunch_index)
{
    int N_rows = NCells;
    int N_cols = tBunch.size();

    int bunchLength = cellTypeBunch.size();

    writeDoubleVectorToFile(tBunch, bunchLength, "data/t_"+ to_string(saved_bunch_index) + ".txt");
    writeIntMatrixToFile(IntTranspose(cellTypeBunch), NCells, cellTypeBunch.size(), "data/Type_"+ to_string(saved_bunch_index) + ".txt");
    writeDoubleMatrixToFile(DoubleTranspose(cellXBunch), NCells, cellXBunch.size(), "data/X_"+ to_string(saved_bunch_index) + ".txt");
    writeDoubleMatrixToFile(DoubleTranspose(cellYBunch), NCells, cellYBunch.size(), "data/Y_"+ to_string(saved_bunch_index) + ".txt");
    writeDoubleMatrixToFile(DoubleTranspose(cellVxBunch), NCells, cellVxBunch.size(), "data/Vx_"+ to_string(saved_bunch_index) + ".txt");
    writeDoubleMatrixToFile(DoubleTranspose(cellVyBunch), NCells, cellVyBunch.size(), "data/Vy_"+ to_string(saved_bunch_index) + ".txt");
    writeDoubleMatrixToFile(DoubleTranspose(cellPhiBunch), NCells, cellPhiBunch.size(), "data/Phi_"+ to_string(saved_bunch_index) + ".txt");
    writeIntMatrixToFile(IntTranspose(cellStateBunch), NCells, cellStateBunch.size(), "data/State_"+ to_string(saved_bunch_index) + ".txt");
    writeFitnessToFile(cellFitnessBunch, NCells, bunchLength, "data/Fit_"+ to_string(saved_bunch_index) + ".txt");
    

}

void writeVecOfVecsUnequal(const std::vector<std::vector<double>>& vec_of_vecs,
                           const int N_desired, const std::string& filename) {
    // Open the file to write
    std::ofstream outfile(filename);

    if (outfile.is_open()) {
        // Loop over each vector in vec_of_vecs up to N_desired inner vectors
        for (size_t i = 0; i < vec_of_vecs.size() && i < N_desired; ++i) {
            const auto& inner_vec = vec_of_vecs[i];
            
            if (!inner_vec.empty()) { // Only write non-empty vectors
                // Write each element in the inner vector
                for (size_t j = 0; j < inner_vec.size(); ++j) {
                    outfile << std::fixed << std::setprecision(6) << inner_vec[j];
                    if (j < inner_vec.size() - 1) { // Add a comma if it's not the last element
                        outfile << ",";
                    }
                }
            }
            outfile << "\n"; // Newline
        }
        outfile.close(); // Close the file
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}
//////////////////////////////////////////////////////////////////////////
////////////////////////////// FUNCTIONS /////////////////////////////////
//////////////////////////////////////////////////////////////////////////