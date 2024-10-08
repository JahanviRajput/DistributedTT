from opti import Opti

try:
    import sys
    sys.path.append("../")
    from Methods import subset_submod_pts
    with_module = True
except Exception as e:
    with_module = False


class Optissptsfl(Opti):
    def __init__(self, name='SSPTS', *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def opts(self, k=100, k_gd=1, lr=5.E-2, r=5, P=None,
             seed=0, subset_size = 10, sub_fun = 'FL'):
        self.opts_k = k
        self.opts_k_gd = k_gd
        self.opts_lr = lr
        self.opts_r = r
        self.opts_P = P
        self.opts_seed = seed
        self.subset_size = subset_size
        self.sub_fun = sub_fun

    def _init(self):
        if not with_module:
            self.err = 'Need "SSPTS" module'
            return

    def _optimize(self):
        subset_submod_pts(self.f_batch, self.d, self.n[0], self.m_max, k=self.opts_k, P=self.opts_P,
         k_gd=self.opts_k_gd, lr=self.opts_lr, r=self.opts_r, is_max=self.is_max,
            seed=self.opts_seed, sub_fun = self.sub_fun)

        
        

