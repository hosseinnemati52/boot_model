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
void simulationDataReader(int* NSitesPtr, double* LxPtr, double* LyPtr, double* AlphaPtr, double* KbPtr, double* TemPtr,  int* NumCellsPtr, \
                          double* AvgCellAreaPtr, double* LambdaPtr, long* maxMCStepsPtr, int* samplesPerWritePtr, \
                          int* printingTimeIntervalPtr, int* numLinkedListPtr, string* initConfigPtr);

std::vector<double> parseVector(const std::string& str);

std::vector<std::vector<double>> parseMatrix(const std::string& str);

void trim(std::string& str);

void readConfigFile(const std::string& filename, Config& config);

void initializer_specific(const int N_UpperLim, int* NCellsPtr, vector<int>& NCellsPerType,
                 const vector<double> typeR0, const vector<double> typeR2PI, 
                 vector<int>& cellType, vector<double>& cellX, vector<double>& cellY, 
                 vector<double>& cellVx, vector<double>& cellVy, 
                 vector<double>& cellPhi, vector<int>& cellState, vector<double>& cellTheta,
                 vector<vector<double>>& cellFitness, const vector<double>& typeFit0, std::mt19937 &mt_rand,
                 const double G1Border, const double Fit_Th_G0, const double Fit_Th_Apop, const vector<double>& typeSigmaFit);

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
//////////////////////////////////////////////////////////////////////////
////////////////////////////// PROTOTYPES ////////////////////////////////
//////////////////////////////////////////////////////////////////////////

int main()
{

    /////////////////// FOLDERS NAMES ////////////////////
    // std::string dataFolderName = "data";
    std::string initOfInitFolderName = "init_of_init";
    std::string initFolderName = "init";
    // std::string mainResumeFolderName = "main_resume";
    // std::string backupResumeFolderName = "backup_resume";
    // std::string loadFolderName;
    /////////////////// FOLDERS NAMES ////////////////////

    /////////////////// MAKING SUB DIRECTORIES /////////////////
    // This block is for windows:
    // mkdir(dataFolderName.c_str()); //making data folder
    // mkdir(initFolderName.c_str()); //making init folder
    // mkdir(mainResumeFolderName.c_str()); //making main_resume folder
    // mkdir(backupResumeFolderName.c_str()); //making backup_resume folder

    // This block is for Linux:
    // mkdir(dataFolderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making data folder
    mkdir(initOfInitFolderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making init_of_init folder
    mkdir(initFolderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making init folder
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
    // mt_rand_seed = 1242276940;

    //Seeding
    mt_rand.seed(mt_rand_seed);
    unsigned long MT_MAX = mt_rand.max();
    unsigned long MT_MIN = mt_rand.min();
    // saving initial random seed
    ofstream randSeedInit;
    randSeedInit.open(initOfInitFolderName + "/" + "randSeedInit.csv");
    randSeedInit << mt_rand_seed;
    randSeedInit.close(); // random seed saved
    

    // saving initial random generator
    std::ofstream randStateInit(initOfInitFolderName + "/" + "randStateInit.csv");
    randStateInit << mt_rand;
    randStateInit.close();
    /////////////////// RANDOM GENERATOR SEEDING /////////////////
    
    
    ///////////////// INITIALIZATION ////////////////////
    int saved_bunch_index = 1;
    
    // initializer(N_UpperLim, &NCells, NCellsPerType,
    //              typeR0, typeR2PI, 
    //              cellType, cellX, cellY, 
    //              cellVx, cellVy, 
    //              cellPhi, cellState, cellTheta,
    //              cellFitness, typeFit0, mt_rand);

    initializer_specific(N_UpperLim, &NCells, NCellsPerType,
                 typeR0, typeR2PI, 
                 cellType, cellX, cellY, 
                 cellVx, cellVy, 
                 cellPhi, cellState, cellTheta,
                 cellFitness, typeFit0, mt_rand,
                 G1Border, Fit_Th_G0, Fit_Th_Apop, typeSigmaFit);

    R_Area_calc(N_UpperLim, NCells, NTypes,
                typeR0, typeR2PI, 
                cellType,
                cellPhi, cellState,
                cellR, cellArea); // cellR and cellArea are calculated. The others are const in the function.
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

    // for identifying immediate neighbors
    // int NN_force[N_UpperLim][N_UpperLim];
    // int NN_game[N_UpperLim][N_UpperLim];
    vector<vector<int>> NN_force(N_UpperLim, vector<int>(N_UpperLim));
    vector<vector<int>> NN_game(N_UpperLim, vector<int>(N_UpperLim));
    for (cellC_1 = 0; cellC_1 < N_UpperLim; cellC_1++)
    {
        for (cellC_2 = 0; cellC_2 < N_UpperLim; cellC_2++)
        {
            NN_force[cellC_1][cellC_2] = 0;
            NN_game[cellC_1][cellC_2] = 0;
        }
    }
    
    
    
    // Update Auxiliary properties vectors
    vector<vector<double>> cellFitnessUpdated(N_UpperLim, vector<double>(2)); // This stores the updated values of fitness of cells (Real and Imaginary parts).
    vector<double> cellPhiUpdated(N_UpperLim); // Updated phases in cell cycle, \in [0, 2*pi]
    vector<double> cellRUpdated(N_UpperLim); // Updated Radius of the cells
    vector<int> cellStateUpdated(N_UpperLim); // Cell state Updated values: {cycling:CYCLING_STATE, G1_arr:G1_ARR_STATE, G0:G0_STATE, differentiated:DIFF_STATE, apop:APOP_STATE, does not exist: 0}

    vector<double> cellXUpdated(N_UpperLim); // Updated X
    vector<double> cellYUpdated(N_UpperLim); // Updated Y

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
    // Update Auxiliary properties vectors

    int debug_var;

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
            
            cellRUpdated[cellC_1] = cellR[cellC_1];

        }
        // setting forces, and fluxes to zero
        


        // These two for loops are for calculating center-to-center force terms
        for (cellC_1 = 0; cellC_1 < NCells; cellC_1++) // loop on cellC_1
        {
            
            cellType_1 = cellType[cellC_1];

            for (cellC_2 = cellC_1 + 1 ; cellC_2 < NCells; cellC_2++) // loop on cellC_2, for interactions (force and game)
            {

                cellType_2 = cellType[cellC_2];

                delta_x = cellXUpdated[cellC_2] - cellXUpdated[cellC_1];
                delta_y = cellYUpdated[cellC_2] - cellYUpdated[cellC_1];
                distance = pow( (delta_x * delta_x + delta_y * delta_y) , 0.5);

                R_eq =   R_eq_coef * (cellRUpdated[cellC_2] + cellRUpdated[cellC_1]);
                R_cut_force = R_cut_coef_force * (cellRUpdated[cellC_2] + cellRUpdated[cellC_1]);

                if (distance < R_eq )
                {
                    F = typeTypeF_rep_max[cellType_1][cellType_2] * (distance - R_eq) / R_eq;
                } else if (distance < R_cut_force )
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
                else
                {
                    F = 0.0;
                }

                

                FaddTermX =  F * (delta_x / distance);
                FaddTermY =  F * (delta_y / distance);

                cellFx[cellC_1] += FaddTermX;
                cellFy[cellC_1] += FaddTermY;

                cellFx[cellC_2] -= FaddTermX;
                cellFy[cellC_2] -= FaddTermY;


            } // the end of "for (cellC_2 = cellC_1 + 1 ; cellC_2 < NCells; cellC_2++)"


            // Here, the V(t+dt) for cellC_1 is calculated
            cellVx[cellC_1] =   cellFx[cellC_1]  / typeGamma[cellType_1] ;
            cellVy[cellC_1] =   cellFy[cellC_1]  / typeGamma[cellType_1] ;
            // Here, the V(t+dt) for cellC_1 is calculated


        } // the end of "for (cellC_1 = 0; cellC_1 < NCells; cellC_1++)"
        // These two for loops are for calculating center-to-center force terms



        for (cellC_1 = 0; cellC_1 < NCells; cellC_1++)
        {
            cellType_1 = cellType[cellC_1];

            // Updating the original vectors; these also work for dead cells
            cellX[cellC_1] = cellXUpdated[cellC_1];
            cellY[cellC_1] = cellYUpdated[cellC_1];

        } // the end of "for (cellC_1 = 0; cellC_1 < NCells; cellC_1++)" for State update and cell division operations.


        t += dt;
    }
    /////// SIMULATION LOOP /////////////

    writeIntVectorToFile(cellType, NCells, initFolderName+"/Type_init.txt");
    writeDoubleVectorToFile(cellX, NCells, initFolderName+"/X_init.txt");
    writeDoubleVectorToFile(cellY, NCells, initFolderName+"/Y_init.txt");
    writeDoubleVectorToFile(cellVx, NCells, initFolderName+"/Vx_init.txt");
    writeDoubleVectorToFile(cellVy, NCells, initFolderName+"/Vy_init.txt");
    writeDoubleVectorToFile(cellPhi, NCells, initFolderName+"/Phi_init.txt");
    writeDoubleVectorToFile(cellTheta, NCells, initFolderName+"/Theta_init.txt");
    writeIntVectorToFile(cellState, NCells, initFolderName+"/State_init.txt");
    // writeDoubleVectorToFile(cellR, NCells, loadFolderName+"/R_LS.txt");
    writeDoubleMatrixToFile(cellFitness, NCells, 2,  initFolderName+"/Fit_init.txt");
    


    
    // system("python3 dataZipper.py");



    return 0;
}


//////////////////////////////////////////////////////////////////////////
////////////////////////////// FUNCTIONS /////////////////////////////////
//////////////////////////////////////////////////////////////////////////

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

void initializer_specific(const int N_UpperLim, int* NCellsPtr, vector<int>& NCellsPerType,
                 const vector<double> typeR0, const vector<double> typeR2PI, 
                 vector<int>& cellType, vector<double>& cellX, vector<double>& cellY, 
                 vector<double>& cellVx, vector<double>& cellVy, 
                 vector<double>& cellPhi, vector<int>& cellState, vector<double>& cellTheta,
                 vector<vector<double>>& cellFitness, const vector<double>& typeFit0, std::mt19937 &mt_rand,
                 const double G1Border, const double Fit_Th_G0, const double Fit_Th_Apop, const vector<double>& typeSigmaFit)
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


    

    int typeInd = 0;
    double A_min = PI * typeR0[typeInd] * typeR0[typeInd];
    double A_max = PI * typeR2PI[typeInd] * typeR2PI[typeInd];

    // specific phi and F for initialization
    random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
    double SG2M_frac = 0.2 + 0.05 * (2*random_float-1);

    random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
    double G0Diff_frac = 0.2 + 0.05 * (2*random_float-1);

    double G1_frac = 1.0 - SG2M_frac - G0Diff_frac;

    int N_WT_SG2M = (int)(SG2M_frac * NCellsPerType[0]);
    int N_WT_G0Diff = (int)(G0Diff_frac * NCellsPerType[0]);
    int N_WT_G1 = NCellsPerType[0] - N_WT_SG2M - N_WT_G0Diff;
    int N_WT_G1_free = (int)((0.27/0.6) * N_WT_G1);
    int N_WT_G1_bound = N_WT_G1 - N_WT_G1_free;
    // specific phi and F for initialization

    int cellC = 0;

    while (cellC < NCellsPerType[0])
    {   
        cellType[cellC] = typeInd;

        if (cellC < N_WT_G1) // The ones in G1
        {
            if (cellC < N_WT_G1_free)
            {
                // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
                cellPhi[cellC] = random_float * 0.95 * (G1Border * 2*PI);
            }
            else if (cellC < N_WT_G1_free+N_WT_G1_bound)
            {
                // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
                cellPhi[cellC] = 0.9 * (G1Border * 2*PI) + random_float * 0.05 * (G1Border * 2*PI);
            }
            
            cellState[cellC] = CYCLING_STATE; 

            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            cellFitness[cellC][0] = typeFit0[typeInd] + (2*random_float-1) * typeSigmaFit[0];
            cellFitness[cellC][1] = 0.0;

        } else if (cellC < N_WT_G1+N_WT_SG2M) // The ones in SG2M
        {
            // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            cellPhi[cellC] = G1Border * 2*PI + random_float * (2*PI - G1Border * 2*PI);

            cellState[cellC] = CYCLING_STATE; 

            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            cellFitness[cellC][0] = typeFit0[typeInd] +typeSigmaFit[0]+ (2*random_float-1) * typeSigmaFit[0];
            cellFitness[cellC][1] = 0.0;

        } else // The ones in G0/Diff
        {
            // random_float = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            cellPhi[cellC] = random_float * (G1Border * 2*PI);

            cellState[cellC] = G0_STATE;     

            random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
            cellFitness[cellC][0] = Fit_Th_Apop + random_float * (Fit_Th_G0 - Fit_Th_Apop);
            cellFitness[cellC][1] = 0.0;
        }

        

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

        random_float = ((long double)(mt_rand())-MT_MIN)/((long double)MT_MAX-MT_MIN);
        cellFitness[cellC][0] = typeFit0[typeInd] + (2*random_float-1) * 0.5 * (typeFit0[typeInd]-Fit_Th_Apop);
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
                if ((x-cellX[j])*(x-cellX[j]) + (y-cellY[j])*(y-cellY[j]) < pow(0.50 * (cellR[cellC]+cellR[j]), 2)  )
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
    // mkdir("init_of_init", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making backup_resume folder

    writeIntVectorToFile(cellType, NCells, "init_of_init/Type_init.txt");
    writeDoubleVectorToFile(cellX, NCells, "init_of_init/X_init.txt");
    writeDoubleVectorToFile(cellY, NCells, "init_of_init/Y_init.txt");
    writeDoubleVectorToFile(cellVx, NCells, "init_of_init/Vx_init.txt");
    writeDoubleVectorToFile(cellVy, NCells, "init_of_init/Vy_init.txt");
    writeDoubleVectorToFile(cellPhi, NCells, "init_of_init/Phi_init.txt");
    writeIntVectorToFile(cellState, NCells, "init_of_init/State_init.txt");
    // writeDoubleVectorToFile(cellR, NCells, "init/R_init.txt");
    writeDoubleVectorToFile(cellTheta, NCells, "init_of_init/Theta_init.txt");
    writeDoubleMatrixToFile(cellFitness, NCells, 2,  "init_of_init/Fit_init.txt");
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
    // mkdir("init_of_init", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); //making backup_resume folder

    writeIntVectorToFile(cellType, NCells, "init_of_init/Type_init.txt");
    writeDoubleVectorToFile(cellX, NCells, "init_of_init/X_init.txt");
    writeDoubleVectorToFile(cellY, NCells, "init_of_init/Y_init.txt");
    writeDoubleVectorToFile(cellVx, NCells, "init_of_init/Vx_init.txt");
    writeDoubleVectorToFile(cellVy, NCells, "init_of_init/Vy_init.txt");
    writeDoubleVectorToFile(cellPhi, NCells, "init_of_init/Phi_init.txt");
    writeIntVectorToFile(cellState, NCells, "init_of_init/State_init.txt");
    // writeDoubleVectorToFile(cellR, NCells, "init/R_init.txt");
    writeDoubleVectorToFile(cellTheta, NCells, "init_of_init/Theta_init.txt");
    writeDoubleMatrixToFile(cellFitness, NCells, 2,  "init_of_init/Fit_init.txt");
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
//////////////////////////////////////////////////////////////////////////
////////////////////////////// FUNCTIONS /////////////////////////////////
//////////////////////////////////////////////////////////////////////////