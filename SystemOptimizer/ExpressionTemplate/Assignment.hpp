#pragma
#include "MathFunc.hpp"
template <typename Kernel, int start, int end>
class Assignment{
    public:
        static void run(Kernel & a){
            a.assign(start);
            Assignment<Kernel, start + 1, end>::run(a);
        }
};
template <typename Kernel, int end>
class Assignment<Kernel, end, end>{
    public:
        static void run(Kernel & a){}
};

template <typename Dst, typename Src, typename Func = Assign<typename Dst::Scalar>>
class general_assignment {
    public:
    general_assignment(Dst& dst, Src & src, Func func = Func()):m_d(dst),m_s(src),m_func(func){}
    void assign(int Index){
        m_func(m_d.getR(Index), m_s.get(Index));
    }
    private:
    Dst m_d;
    Src m_s;
    Func m_func;
};

template <typename Dst, typename Src>
inline void call_assignment(Dst & dst, Src & src){
    using EDstType = evaluator<Dst>;
    using ESrcType = evaluator<Src>;
    EDstType EDst(dst);
    ESrcType ESrc(src);
    using Kernel = general_assignment<EDstType, ESrcType>;
    Kernel kernel(EDst, ESrc);
    Assignment<Kernel, 0, Trait<Dst>::Size>::run(kernel);
}