"""
For tests
"""
import time
from model_gm import ModelGM, sample_gm
from em import EM
from mm import MM
from dmm import DMM




if __name__ == '__main__':
    comp = 2
    model = ModelGM(w=[0.5, 0.5], x=[-1, 1], std=1)
    sample = sample_gm(model, 10000)
    means = model.mean_rv()


    num_comps = 3
    dmm = DMM(k=num_comps, sigma=1)
    mm = MM(k=num_comps, sigma=1)
    em = EM(k=num_comps, sigma=1)
    esti_em = em.estimate(sample)
    print(esti_em.mean_rv(), means.dist_w1(esti_em.mean_rv()))
    esti_dmm = dmm.estimate(sample)
    print(esti_dmm, means.dist_w1(esti_dmm))
    # print(mm.estimate(sample))

    # dmm = DMM(k=2)
    # print(dmm.estimate(sample))
