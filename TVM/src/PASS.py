import tvm
from tvm import relay

@tvm.ir.transform.module_pass(opt_level=2)
def transform(mod, ctx):
    tp = relay.TensorType((10,), "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("abs")
    func = relay.Function([x], relay.abs(x))
    new_mod = tvm.IRModule({gv: func})
    new_mod.update(mod)
    # print(mod)
    import pdb
    pdb.set_trace()
    return new_mod

module_pass = transform
print(module_pass)
mod = tvm.IRModule()
mod = module_pass(mod)
# assert isinstance(module_pass, transform.ModulePass)
# assert module_pass.info.opt_level == 2