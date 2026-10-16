// Windows prototype using the official HiGHS C ABI from a C++ process.
// The input CSC matrix is built by the canonical Python SANOS implementation.
// This intentionally isolates language/interface overhead from matrix assembly.
#define NOMINMAX
#include <windows.h>
#include <interfaces/highs_c_api.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
template<class T> std::vector<T> read_vector(std::ifstream& in, std::size_t n) {
    std::vector<T> out(n);
    if (!in.read(reinterpret_cast<char*>(out.data()), n*sizeof(T))) throw std::runtime_error("Truncated input");
    return out;
}
template<class T> T symbol(HMODULE dll, const char* name) {
    auto ptr = GetProcAddress(dll,name);
    if (!ptr) throw std::runtime_error(std::string("Missing HiGHS symbol: ")+name);
    return reinterpret_cast<T>(ptr);
}
void check_status(HighsInt status,const char* expression) {
    if(status<0) throw std::runtime_error(std::string("HiGHS API error: ")+expression);
}
#define check(expression) check_status(expression,#expression)

int main(int argc,char** argv) try {
    if(argc!=6) throw std::runtime_error("Usage: incremental_highs DLL INPUT {cold|warm} SOLUTIONS TIME_LIMIT");
    static_assert(sizeof(HighsInt)==4,"Binary format requires 32-bit HighsInt");
    std::string mode=argv[3];
    if(mode!="cold" && mode!="warm") throw std::runtime_error("Invalid mode");
    std::ifstream in(argv[2],std::ios::binary);
    auto magic=read_vector<char>(in,8);
    if(std::string(magic.begin(),magic.end())!="XTXLP001") throw std::runtime_error("Invalid input format");
    auto sizes=read_vector<std::uint64_t>(in,4);
    for(auto n:sizes) if(n>static_cast<std::uint64_t>(std::numeric_limits<HighsInt>::max())) throw std::runtime_error("Input too large");
    HighsInt nc=sizes[0],nr=sizes[1],nz=sizes[2],nf=sizes[3];
    auto start=read_vector<HighsInt>(in,nc+1), index=read_vector<HighsInt>(in,nz);
    auto value=read_vector<double>(in,nz), lower=read_vector<double>(in,nr);
    std::vector<std::vector<double>> costs, uppers;
    for(HighsInt f=0;f<nf;++f) { costs.push_back(read_vector<double>(in,nc)); uppers.push_back(read_vector<double>(in,nr)); }
    HMODULE dll=LoadLibraryA(argv[1]);
    if(!dll) throw std::runtime_error("Could not load official HiGHS DLL");
#define API(name) auto name=symbol<decltype(&Highs_##name)>(dll,"Highs_" #name)
    API(create); API(destroy); API(passLp); API(run); API(version);
    API(setBoolOptionValue); API(setIntOptionValue); API(setDoubleOptionValue); API(setStringOptionValue);
    API(changeRowsBoundsByRange); API(changeColsCostByRange); API(getIntInfoValue);
    API(getModelStatus); API(getObjectiveValue); API(getSolution); API(getRunTime);
#undef API
    std::ofstream solutions(argv[4],std::ios::binary);
    if(!solutions) throw std::runtime_error("Cannot write solutions");
    std::vector<double> col_lower(nc,0),col_upper(nc,1e30),x(nc);
    void* h=nullptr;
    std::cout<<std::setprecision(17);
    for(HighsInt f=0;f<nf;++f) {
        auto total=Clock::now(); auto stage=total;
        if(mode=="cold" || f==0) {
            h=create();
            check(setBoolOptionValue(h,"output_flag",std::getenv("XTX_HIGHS_VERBOSE")?1:0));
            check(setIntOptionValue(h,"threads",1));
            check(setIntOptionValue(h,"random_seed",0));
            check(setIntOptionValue(h,"simplex_strategy",1));
            check(setStringOptionValue(h,"solver","simplex"));
            check(setStringOptionValue(h,"presolve","off"));
            check(setDoubleOptionValue(h,"time_limit",std::stod(argv[5])));
            check(setDoubleOptionValue(h,"primal_feasibility_tolerance",1e-7));
            check(setDoubleOptionValue(h,"dual_feasibility_tolerance",1e-7));
            check(passLp(h,nc,nr,nz,1,1,0,costs[f].data(),col_lower.data(),col_upper.data(),
                         lower.data(),uppers[f].data(),start.data(),index.data(),value.data()));
        } else {
            check(changeRowsBoundsByRange(h,0,nr-1,lower.data(),uppers[f].data()));
            check(changeColsCostByRange(h,0,nc-1,costs[f].data()));
        }
        check(setDoubleOptionValue(h,"time_limit",getRunTime(h)+std::stod(argv[5])));
        double update_seconds=seconds(stage); stage=Clock::now();
        auto run_status=run(h);
        double solve_seconds=seconds(stage); stage=Clock::now();
        auto status=getModelStatus(h); auto objective=getObjectiveValue(h);
        HighsInt iterations=0;
        if(status==7) {
            check(getIntInfoValue(h,"simplex_iteration_count",&iterations));
            check(getSolution(h,x.data(),nullptr,nullptr,nullptr));
        } else { std::fill(x.begin(),x.end(),0.);objective=0.; }
        double extraction_seconds=seconds(stage),total_seconds=seconds(total);
        solutions.write(reinterpret_cast<char*>(&status),sizeof(status));
        solutions.write(reinterpret_cast<char*>(&objective),sizeof(objective));
        solutions.write(reinterpret_cast<char*>(x.data()),nc*sizeof(double));
        std::cout<<"{\"mode\":\"cpp_"<<mode<<"\",\"version\":\""<<version()
                 <<"\",\"frame\":"<<f<<",\"model_status\":"<<status<<",\"iterations\":"<<iterations
                 <<",\"run_api_status\":"<<run_status
                 <<",\"update_seconds\":"<<update_seconds<<",\"solve_seconds\":"<<solve_seconds
                 <<",\"extraction_seconds\":"<<extraction_seconds<<",\"total_seconds\":"<<total_seconds
                 <<",\"objective\":"<<objective<<"}"<<std::endl;
        if(mode=="cold") { destroy(h); h=nullptr; }
    }
    if(h) destroy(h);
    FreeLibrary(dll);
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
