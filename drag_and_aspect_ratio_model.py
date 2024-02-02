#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from decimal import *
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.constants import g, pi, gas_constant
from scipy.optimize import least_squares, differential_evolution
from scipy.linalg import svd

getcontext().prec = 34

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['legend.frameon'] = False

class dragAndAspectRatioModel:
	def __init__(self, inputs:dict):
		self.cwd = inputs['cwd']
		self.egReal = inputs['eg_real']
		self.egFolder = inputs['eg_folder']
		self.pReal = inputs['p_real']
		self.pFolder = inputs['p_folder']
		self.temp = inputs['temp']
		self.minChord = inputs['minChord']
		self.chordIncrement = inputs['chordIncrement']
		self.nClasses = inputs['nClasses']
		self.rhol = inputs['rhol']
		self.mul = inputs['mul']
		self.sigma = inputs['sigma']
		self.UL = inputs['UL']
		self.rRatio = inputs['rRatio']
		self.difAreas = inputs['difAreas']
		self.aspectRatioModel = inputs['aspectRatioModel']
		self.MW = inputs['MW']
		self.plotting = inputs['plotting']
		self.nGasHoldups = len(inputs['eg_folder'])
		self.nPressures = len(inputs['p_folder'])
	def Wellek(self, Eo, beta0, beta1):
		E = 1.0/(1.0 + beta0*Eo**beta1)
		return E
	def BesagniDeen(self, Re, Eo, beta0, beta1):
		E = 1.0/((1.0 + beta0*Re*Eo)**beta1)
		return E
	def BesagniDeenRevised(self, Re, Eo, beta0, beta1):
		E = 1.0/(1.0 + beta0*(Re*Eo)**beta1)
		return E
	def chordToDiameter(self, cbi, usi, rhog, beta0, beta1):
		Re = (self.rhol*usi*0.0015*cbi)/self.mul
		Eo = ((self.rhol - rhog)*g*(0.0015*cbi)**2)/self.sigma
		
		if(self.aspectRatioModel == "Wellek"):
			E = self.Wellek(Eo, beta0, beta1)
		elif(self.aspectRatioModel == "BesagniDeen"):
			E = self.BesagniDeen(Re, Eo, beta0, beta1)
		elif(self.aspectRatioModel == "BesagniDeenRevised"):
			E = self.BesagniDeenRevised(Re, Eo, beta0, beta1)
		else:
			E = self.BesagniDeenRevised(Re, Eo, beta0, beta1)
		
		dbi = 1.5*cbi*(E**(-2.0/3.0))
		
		relError = 1.0
		relTol = 1.0e-8
		
		while(relError > relTol):
			dbi_prev = dbi
			Re = (self.rhol*usi*0.001*dbi_prev)/self.mul
			Eo = ((self.rhol - rhog)*g*(0.001*dbi_prev)**2)/self.sigma

			if(self.aspectRatioModel == "Wellek"):
				E = self.Wellek(Eo, beta0, beta1)
			elif(self.aspectRatioModel == "BesagniDeen"):
				E = self.BesagniDeen(Re, Eo, beta0, beta1)
			elif(self.aspectRatioModel == "BesagniDeenRevised"):
				E = self.BesagniDeenRevised(Re, Eo, beta0, beta1)
			else:
				E = self.BesagniDeenRevised(Re, Eo, beta0, beta1)
			
			dbi = 1.5*cbi*(E**(-2.0/3.0))
			relError = abs((dbi - dbi_prev)/dbi)

		return dbi

	def createClasses(self, df):
		rows = len(df)
		
		count = np.zeros(self.nClasses)
		cbr_num = np.zeros(self.nClasses)
		ubr_num = np.zeros(self.nClasses)

		for i in range(rows):
			cbri = df['Chord(mm)'][i]
			ubri = df['Velocity(m/s)'][i]
			for j in range(self.nClasses):
				lowerEnd = self.minChord + j*self.chordIncrement
				upperEnd = self.minChord + (j+1)*self.chordIncrement
				if(cbri > lowerEnd and cbri < upperEnd):
					count[j] = count[j] + 1
					cbr_num[j] = cbr_num[j] + cbri
					ubr_num[j] = ubr_num[j] + ubri

		classes = np.zeros([self.nClasses, 3])

		for k in range(self.nClasses):
			classes[k,0] = cbr_num[k]/count[k]
			classes[k,1] = ubr_num[k]/count[k]
			classes[k,2] = count[k]

		return classes

	def radialAverage(self, classes, P, eg_folder, eg, rhog, beta0, beta1):
		Cbnums = np.zeros(self.nClasses)
		Cbdens = np.zeros(self.nClasses)
		Ubnums = np.zeros(self.nClasses)
		Ubdens = np.zeros(self.nClasses)

		nrR = len(self.rRatio)
		
		for i in range(nrR):
			key = 'P' + str(P) + '-eG' + str(eg_folder) + '-rR' + self.rRatio[i]
			arr = classes[key]
			for j in range(self.nClasses):
				Cbnums[j] = Cbnums[j] + (arr[j,0]**3)*arr[j,2]*self.difAreas[i]
				Cbdens[j] = Cbdens[j] + arr[j,2]*self.difAreas[i]

				Ubnums[j] = Ubnums[j] + arr[j,1]*(arr[j,0]**3)*arr[j,2]*self.difAreas[i]
				Ubdens[j] = Ubdens[j] + (arr[j,0]**3)*arr[j,2]*self.difAreas[i]

		results = np.zeros([self.nClasses, 3])
		for k in range(self.nClasses):
			Ubi = Ubnums[k]/Ubdens[k]
			Usi = Ubi - self.UL/(1.0 - eg)
			
			Cbi = (Cbnums[k]/Cbdens[k])**(1.0/3.0)
			dbi = self.chordToDiameter(Cbi, Usi, rhog, beta0, beta1)
			
			Re = (self.rhol*Usi*0.001*dbi)/self.mul
			Eo = ((self.rhol - rhog)*g*(0.001*dbi)**2)/self.sigma

			if(self.aspectRatioModel == "Wellek"):
				E = self.Wellek(Eo, beta0, beta1)
			elif(self.aspectRatioModel == "BesagniDeen"):
				E = self.BesagniDeen(Re, Eo, beta0, beta1)
			elif(self.aspectRatioModel == "BesagniDeenRevised"):
				E = self.BesagniDeenRevised(Re, Eo, beta0, beta1)
			else:
				E = self.BesagniDeenRevised(Re, Eo, beta0, beta1)

			Cdi = (4.0/3.0)*(self.rhol - rhog)/self.rhol*(g*(0.001*dbi)*E**(2.0/3.0))/(Usi**2)

			results[k,0] = dbi
			results[k,1] = Usi
			results[k,2] = Cdi

		return results
	def readFileContents(self, filenames):
		classes = {}
		for fname in filenames:
			_, f = os.path.split(fname)
			key, _ = os.path.splitext(f)
			df = pd.read_csv(fname, delimiter='\t')
			classes[key] = self.createClasses(df)
		return classes
	def generateModelData(self, beta0, beta1, plotting=False):
		drag_data = pd.DataFrame(data = None)
		bins = {}
		m_data = []
		mdict = {}
		if(plotting == True):
			plt.figure(figsize=[6, 5*self.nGasHoldups])
		for i in range(self.nGasHoldups):
			if(plotting == True):
				plt.subplot(1, nGasHoldups, i+1)
			for j in range(self.nPressures):
				eg = self.egReal[self.nGasHoldups*i +j]
				T = self.temp[self.nGasHoldups*i + j]
				P = self.pReal[self.nGasHoldups*i + j]
				rhog = (P*1e6*self.MW)/(gas_constant*T)
				pathToFiles = self.cwd + '/P = ' + str(self.pFolder[j]) + ' MPa/eG = ' + str(self.egFolder[i])
				filenames = glob.glob(os.path.join(pathToFiles, '*.txt'))
				bins = self.readFileContents(filenames)
				globalAverages = self.radialAverage(bins, self.pFolder[j], self.egFolder[i], self.egReal[i], rhog, beta0, beta1)
				dia = globalAverages[:,0]
				dragCoeff = globalAverages[:,2]
				if(plotting  == True):
					plt.scatter(dia, dragCoeff, label='P = ' + str(pFolder[j]) + 'MPa')
				keystring = 'P' + str(int(1000*self.pFolder[j])) + '-eG' + str(int(10000*self.egFolder[i]))
				Re = (self.rhol*globalAverages[:,1]*0.001*globalAverages[:,0])/self.mul
				Eo = ((self.rhol - rhog)*g*(0.001*globalAverages[:,0])**2)/self.sigma
				drag_data[keystring + '-dbi'] = globalAverages[:,0]
				drag_data[keystring + '-Usi'] = globalAverages[:,1]
				drag_data[keystring + '-Cdi'] = globalAverages[:,2]
				drag_data[keystring + '-pPrime'] = (P/0.10)*np.ones(self.nClasses)
				drag_data[keystring + '-eg'] = eg*np.ones(self.nClasses)
				drag_data[keystring + '-Re'] = Re
				drag_data[keystring + '-Eo'] = Eo

				pNorm = P/(0.10*Eo)

				mdict['Re'] = Re
				mdict['Eo'] = Eo
				mdict['eg'] = eg*np.ones(self.nClasses)
				mdict['pNorm'] = pNorm
				mdict['Cdi'] = globalAverages[:,2]

				mdf = pd.DataFrame(mdict)
				m_data.append(mdf)
		
			if(plotting == True):
				plt.xlabel(r'bubble diameter (mm)')
				plt.ylabel(r'drag coefficient (-)')
				plt.legend()

		if(plotting == True):
			plt.savefig('dragPlot.png', dpi=300)
			plt.close()

		model_data = pd.concat(m_data, ignore_index = True)
		model_data.to_csv('model_data.txt', sep='\t', index=False, na_rep = 'nan')
		drag_data.to_csv('drag_data.txt', sep='\t', index=False, na_rep = 'nan')

		return model_data
	
	def model(self, beta2, beta3, beta4, Re, Eo, eg, pNorm):
		# single bubble drag coefficient
		Cd0 = 24.0/Re
		
		# correction for inertial and buoyant effects
		# ReEo represents the relative magnitudes of form and viscous drag
		fReEo = (1.0 + (Re*Eo)**beta2)

		# pressure correction
		# pNorm is dimensionless pressure
		# normalized by Eotvos and represents
		# the opposing effects of pressure
		# and buoyant forces that strive to
		# maintain a near spherical shape 
		# and render the bubble non-spherical
		# 
		# The ratio can also be viewed as 
		# representing the effects of pressure
		# that are not due to changes in density
		m = beta3*(1.0 - np.exp(-beta4*pNorm))
		
		# swarm correction
		fswarm = (1.0 - eg)**m

		# all effects combined
		Cdi = Cd0*fReEo*fswarm

		return Cdi

	def objFunc(self, betas):
		beta0, beta1, beta2, beta3, beta4 = betas[0], betas[1], betas[2], betas[3], betas[4]
		xydata = self.generateModelData(beta0, beta1)
		
		xydata.dropna(inplace=True)
		xydata.reset_index(drop=True, inplace=True)

		Re = xydata['Re']
		Eo = xydata['Eo']
		eg = xydata['eg']
		pNorm = xydata['pNorm']
		
		Cdi = xydata['Cdi']

		CdiPred = self.model(beta2, beta3, beta4, Re, Eo, eg, pNorm)
			
		residuals = Cdi - CdiPred

		return residuals
	
	def objFunc_de(self, betas):
		residuals = self.objFunc(betas)
		squared_residuals = residuals**2
		result = 0.0
		for resid in squared_residuals:
			result = result + resid
		return result
	def differentialEvo(self, betas0, bounds_):
		params_de = differential_evolution(self.objFunc_de, bounds=bounds_, x0=betas0, strategy='best1exp', disp=True)
		print(params_de.x)
		return params_de
	def fit(self, betas0, bounds_, jacobian, method_, solver=None):
		params_de = self.differentialEvo(betas0, bounds_)
		if(method_ == "trf"):
			trSolver = solver
			params = least_squares(self.objFunc, params_de.x, jac=jacobian, method=method_, tr_solver=trSolver)
		else:
			params = least_squares(self.objFunc, params_de.x, jac=jacobian, method=method_, verbose=2)
		
		plotting = True
		xydata = self.generateModelData(params.x[0], params.x[1], plotting)
		Re = xydata['Re']
		Eo = xydata['Eo']
		eg = xydata['eg']
		pNorm = xydata['pNorm']
		Cdi = xydata['Cdi']

		CdiPred = self.model(params.x[2], params.x[2], params.x[4], Re, Eo, eg, pNorm)

		sr = (Cdi - CdiPred)**2
		ssr = np.sum(sr)
		sst = np.sum((Cdi - np.mean(Cdi))**2)
		rsq = 1.0 - (ssr/sst)

		U, s, Vh = svd(params.jac)
		tol = np.finf(float).eps*s[0]*max(params.jac.shape)
		w = s > tol
		cov = (Vh[w].T/s[w]**2) @ Vh[w]
		perr = np.sqrt(np.diag(cov))/np.sqrt(len(Cdi) - 1)

		print(f"Parameters for Aspect Ratio Model")
		print(f"---------------------------------")
		print(f"beta0 = {params.x[0]:.2f} +/- {perr[0]:.2f}, beta1 = {params.x[1]:.2f} +/- {perr[1]:.2f}")

		print(f"Parameters for Drag Model")
		print(f"-------------------------")
		print(f"beta2 = {params.x[2]:.2f} +/- {perr[2]:.2f}, beta3 = {params.x[3]:.2f} +/- {perr[1]:.2f}")
		print(f"beta3 = {params.x[4]:.2f} +/- {perr[4]:.2f}")

		residuals = Cdi - CdiPred
		
		plt.figure(figsize=[6,5])
		plt.plot(residuals, 'bo')
		plt.xlabel("Observations")
		plt.ylabel("Residuals")
		plt.savefig("residuals.png", dpi=300)
		plt.close()

	def plotBSD(self):
		i = 0
		chords = np.linspace(0, self.minChord + self.nClasses*self.chordIncrement, 3000)
		df_plot = pd.DataFrame(chords, columns=['Chord'])
		plt.figure(figsize=[6,5*self.nClasses])
		for eg in self.egFolder:
			plt.subplot(1, self.nClasses, i+1)
			for p in self.pFolder:
				pathToFiles = self.cwd + '/P = ' + str(p) + ' MPa' + '/eG = ' + str(eg)
				files = sorted(glob.glob(os.path.join(pathToFiles, '*.txt')))
				list_df = []
				for file in files:
					list_df.append(pd.read_csv(file, delimiter='\t'))
				df = pd.concat(list_df)
				shape, loc, scale = lognorm.fit(df['Chords(mm)'])
				cdf = lognorm.cdf(chords, shape, loc, scale)
				pdf = lognorm.pdf(chords, shape, loc, scale)
				df_plot['cdf-P' + str(int(1000*p)) + '-eG' + str(int(10000*eg))] = cdf
				df_plot['pdf-P' + str(int(1000*p)) + '-eG' + str(int(10000*eg))] = pdf
				plt.plot(chords, cdf, lable='P = ' + str(p) + 'MPa')
			
			plt.xlabel(r'bubble chord length (mm)')
			plt.ylabel(r'cumulative probability density')
			plt.legend()
			i = i + 1
		plt.savefig('distribution.png', dpi=300)
		plt.clos()
		df_plot.to_csv('chord-length-cdf.txt', sep='\t', index=False)

