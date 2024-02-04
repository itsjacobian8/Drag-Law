#!/usr/bin/env python3

from drag_and_aspect_ratio_model import *

# data
inputs  = {}
inputs['cwd'] = os.getcwd()
inputs['eg_real'] = [0.092, 0.102, 0.098, 0.104, 0.178, 0.190, 0.187, 0.194, 0.271, 0.242, 0.264, 0.279,0.335, 0.324, 0.331, 0.342]
inputs['eg_folder'] = [0.0987, 0.1873, 0.264, 0.3329]

inputs['p_real'] = [0.14, 0.43, 2.0, 4.2, 0.14, 0.46, 2.0, 4.0, 0.14, 0.46, 2.1, 3.9, 0.14, 0.48, 2.1, 3.8]
inputs['p_folder'] = [0.14, 0.46, 2, 4]

inputs['temp'] = [300.45, 300.235, 301.290, 303.340, 300.96, 300.29, 302.20, 303.340, 301.50, 300.66, 302.93, 304.137, 301.63, 299.543, 303.340, 304.608]

minChord = 0.20
maxChord = 3.00
chordIncrement = 0.20
nClasses = int(round((maxChord - minChord)/chordIncrement))

inputs['minChord'] = minChord
inputs['chordIncrement'] = chordIncrement
inputs['nClasses'] = nClasses

inputs['rhol'] = 998.0
inputs['mul'] = 0.000935
inputs['sigma'] = 0.068
inputs['UL'] = 0.03
inputs['rRatio'] = ['-0.80', '-0.60', '-0.40', '-0.20', '0', '0.20', '0.40', '0.60', '0.80']

R = 0.5*0.1016
A0 = 0.50*pi*(R**2 - (-0.80*R)**2)
A1 = 0.50*pi*((-0.80*R)**2 - (-0.60*R)**2)
A2 = 0.50*pi*((-0.60*R)**2 - (-0.40*R)**2)
A3 = 0.50*pi*((-0.40*R)**2 - (-0.20*R)**2)
A4 = pi*((-0.20*R)**2)
A5 = A3
A6 = A2
A7 = A1
A8 = A0

inputs['difAreas'] = [A0, A1, A2, A3, A4, A5, A6, A7, A8]

inputs['aspectRatioModel'] = "BesagniDeenRevised"
inputs["MW"] = 0.028
inputs["plotting"] = False

if __name__ == '__main__':
	case = dragAndAspectRatioModel(inputs)
	betas0 = np.array([0.163, 0.757, 0.46, 3.96, 0.07])
	bounds = [(-10,10), (-10,10), (-10,10), (-10,10), (-10,10)]
	jacobian = '3-point'
	method = 'trf'
	solver = 'lsmr'
	case.fit(betas0, bounds, jacobian, method, solver)
	case.plotBSD()

