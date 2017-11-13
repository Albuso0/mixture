"""
For tests
"""
import time
from model_gm import ModelGM, sample_gm
from em import EM
from mm import MM
from dmm import DMM
from discrete_rv import wass



if __name__ == '__main__':
    k = 2
    mm = MM(k, sigma=1)
    em = EM(k, sigma=1)
    dmm = DMM(k, sigma=None)

    model = ModelGM(w=[0.5, 0.5], x=[-.5, .5], std=1)
    sample = sample_gm(model, 10000)

    # esti_em = em.estimate(sample)
    # print(wass(esti_em.mean_rv(), model.mean_rv()))
    print(dmm.estimate(sample))
    print(dmm.estimate_online(sample))
    # print(wass(esti_dmm, model.mean_rv()))
    # print(mm.estimate(sample))

    # dmm = DMM(k=2)
    # print(dmm.estimate(sample))
