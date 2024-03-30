# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import pyfeng as pf
import abc

class ModelABC(abc.ABC):
    beta = 1   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr = None, None

    ### Numerical Parameters
    dt = 0.1
    n_path = 10000

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0.0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.beta = beta
        self.intr = intr

    def base_model(self, sigma=None):
        if sigma is None:
            sigma = self.sigma

        if self.beta == 0:
            return pf.Norm(sigma, intr=self.intr)
        elif self.beta == 1:
            return pf.Bsm(sigma, intr=self.intr)
        else:
            raise ValueError(f'0<beta<1 not supported')

    def vol_smile(self, strike, spot, texp=1.0):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        price = self.price(strike, spot, texp, cp=1)
        iv = self.base_model().impvol(price, strike, spot, texp, cp=1)
        return iv

    @abc.abstractmethod
    def price(self, strike, spot, texp=1.0, cp=1):
        """
        Vanilla option price

        Args:
            strike:
            spot:
            texp:
            cp:

        Returns:

        """
        m_bs = pf.Bsm(self.sigma, intr=self.intr)
        c_bs = np.exp(-self.intr * texp) * m_bs.price(strike, spot, texp, cp)
        return c_bs

    def sigma_path(self, texp):
        """
        Path of sigma_t over the time discretization

        Args:
            texp:

        Returns:

        """
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        dt = texp / n_dt
        assert texp == tobs[-1]

        Z_t = np.cumsum(np.random.standard_normal((n_dt, self.n_path)) * np.sqrt(dt), axis=0)
        sigma_t = np.exp(self.vov * (Z_t - self.vov/2 * tobs[:, None]))
        sigma_t = np.insert(sigma_t, 0, np.ones(sigma_t.shape[1]), axis=0)

        return sigma_t

    def intvar_normalized(self, sigma_path):
        """
        Normalized integraged variance I_t = \int_0^T sigma_t dt / (sigma_0^2 T)

        Args:
            sigma_path: sigma path

        Returns:

        """
        #问题3：IT的计算有问题，sigma_t没有平方(已解决)，这里加权已经相当于/T，生成的结果是向量
        weight = np.ones(sigma_path.shape[0])
        weight[[0, -1]] = 0.5
        weight /= weight.sum()
        intvar = np.sum(weight[:, None] * sigma_path**2, axis=0)
        return intvar

class ModelBsmMC(ModelABC):
    """
    MC for Bsm SABR (beta = 1)
    """

    beta = 1.0   # fixed (not used)

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t. 

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        (2) Simulate S_0, ...., S_T.

        Z = np.random.standard_normal()

        (3) Calculate option prices (vector) for all strikes
        '''
        #问题1：必须分割时间递推计算，不能直接从t=0跳到T
        #Generate the paths of sigma_t(重新生成)
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        mean = [0,0]
        cov = [[1, self.rho], [self.rho, 1]]
        W = np.zeros((n_dt, self.n_path))
        Z = np.zeros((n_dt, self.n_path))
        for i in range(self.n_path):
            random_numbers = np.random.multivariate_normal(mean, cov, n_dt)
            W[:,i] = random_numbers[:, 0]
            Z[:,i] = random_numbers[:, 1]

        Z_t = np.cumsum(Z * np.sqrt(self.dt), axis=0)
        sigma_t = np.exp(self.vov * (Z_t - self.vov/2 * tobs[:, None]))
        sigma_t = np.insert(sigma_t, 0, np.ones(sigma_t.shape[1]), axis=0)
        
        vol_path = self.sigma * sigma_t[:-1]
        
        
        W_t = W * np.sqrt(self.dt)
        term_1 = np.cumsum(vol_path * W_t, axis = 0)
        term_2 = np.cumsum(vol_path**2 * self.dt/2, axis = 0)
         
        #Simulate S_0, ...., S_T.
        S_path = spot * np.exp(term_1 - term_2)
        S_T = S_path[-1,:]
        
        #Calculate option prices (vector) for all strikes
        p = np.zeros((len(strike)))
        df = np.exp(-self.intr * texp)
        for i in range(len(strike)):
            p[i] = df * np.mean(np.fmax(S_T - strike[i], 0.0))
        
        return p

class ModelNormMC(ModelBsmMC):
    """
    MC for Normal SABR (beta = 0)
    """

    beta = 0   # fixed (not used)

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t. 

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        (2) Simulate S_0, ...., S_T.

        Z = np.random.standard_normal()

        (3) Calculate option prices (vector) for all strikes
        '''
        #问题2：必须分割时间递推计算，不能直接从t=0跳到T
        #Generate the paths of sigma_t.同理重新生成sigma_t
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        mean = [0,0]
        cov = [[1, self.rho], [self.rho, 1]]
        W = np.zeros((n_dt, self.n_path))
        Z = np.zeros((n_dt, self.n_path))
        for i in range(self.n_path):
            random_numbers = np.random.multivariate_normal(mean, cov, n_dt)
            W[:,i] = random_numbers[:, 0]
            Z[:,i] = random_numbers[:, 1]

        Z_t = np.cumsum(Z * np.sqrt(self.dt), axis=0)
        sigma_t = np.exp(self.vov * (Z_t - self.vov/2 * tobs[:, None]))
        sigma_t = np.insert(sigma_t, 0, np.ones(sigma_t.shape[1]), axis=0)
        
        vol_path = self.sigma * sigma_t[:-1]

        W_t = W * np.sqrt(self.dt)
        
        #Simulate S_0, ...., S_T.
        S_path = spot + np.cumsum(vol_path * W_t, axis = 0)
        S_T = S_path[-1,:]
        
        #Calculate option prices (vector) for all strikes
        df = np.exp(-self.intr * texp)
        p = np.zeros((len(strike)))
        df = np.exp(-self.intr * texp)
        for i in range(len(strike)):
            p[i] = df * np.mean(np.fmax(S_T - strike[i], 0.0))
        return p

#从此处往下已无问题
class ModelBsmCondMC(ModelBsmMC):
    """
    Conditional MC for Bsm SABR (beta = 1)
    """

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t and normalized integrated variance

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 

        (2) Calculate the equivalent spot and volatility of the BS model

        vol = 
        spot_equiv = 

        (3) Calculate option prices (vector) by averaging the BS prices

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        '''
        #Generate the paths of sigma_t and normalized integrated variance
        vol_path = self.sigma_path(texp)
        sigma_t = vol_path[-1, :]
        I_t = self.intvar_normalized(vol_path)

        #Calculate the equivalent spot and volatility of the BS model
        vol = self.sigma * np.sqrt(1-self.rho**2) * np.sqrt(I_t) # Just an example
        spot_equiv = spot * np.exp(self.rho * (sigma_t * self.sigma - self.sigma) / self.vov * np.ones(self.n_path) - self.rho**2 * self.sigma**2 * texp * I_t/2)

        #Calculate option prices (vector) by averaging the BS prices
        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        
        return p


class ModelNormCondMC(ModelNormMC):
    """
    Conditional MC for Bachelier SABR (beta = 0)
    """

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t and normalized integrated variance

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 

        (2) Calculate the equivalent spot and volatility of the Bachelier model

        vol = 
        spot_equiv = 

        (3) Calculate option prices (vector) by averaging the Bachelier prices

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        '''
        #Generate the paths of sigma_t and normalized integrated variance
        vol_path = self.sigma_path(texp)
        sigma_t = vol_path[-1, :]
        I_t = self.intvar_normalized(vol_path)

        #Calculate the equivalent spot and volatility of the Bachelier model
        vol = self.sigma * np.sqrt(1-self.rho**2) * np.sqrt(I_t)  # Just an example
        spot_equiv = spot * np.ones(self.n_path) + self.rho * (sigma_t * self.sigma - self.sigma) / self.vov * np.ones(self.n_path)

        #Calculate option prices (vector) by averaging the Bachelier prices
        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        
        return p