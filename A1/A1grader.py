run_my_solution = False

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A1mysolution import *
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')

    assignmentNumber = '1'

    import subprocess, glob, pathlib
    nb_name = '*A{}*.ipynb'
    # nb_name = '*.ipynb'
    filename = next(glob.iglob(nb_name.format(assignmentNumber)), None)
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         nb_name.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import)):  #  and
            # not isinstance(node, ast.ClassDef) and
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *


    
exec_grade = 0

for func in ['train', 'use', 'rmse']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

            
print('''\nTesting
  X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
  T = (X - 5) * 0.05 + 0.002 * (X - 8)**2
  model = train(X, T, 0.001, 1000, True)
''')

try:
    pts = 20
    X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
    T = (X - 5) * 0.05 + 0.002 * (X - 8)**2
    model = train(X, T, 0.001, 1000, True)
    correct_w = [1.86e-5, 9.92e-1]
    if np.allclose(model['w'].reshape(-1), correct_w, 1e-2):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. w should be')
        print(correct_w)
        print(f'                       Your values are')
        print(model['w'])
except Exception as ex:
    print(f'\n--- 0/{pts} points. train raised the exception\n')
    print(ex)


try:
    pts = 10
    correct_xms = [5.375, 3.352]
    if np.allclose(np.hstack((model['Xmeans'], model['Xstds'])), correct_xms, 1e-2):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Xmeans and Xstds are correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Xmeans, Xstds should be')
        print(correct_xms)
        print(f'                       Your values are')
        print(model['Xmeans'], model['Xstds'])

except Exception as ex:
    print(f'\n--- 0/{pts} points. model key indexing raised the exception\n')
    print(ex)


try:
    pts = 10
    correct_Tms = [0.055, 0.1414]
    if np.allclose(np.hstack((model['Tmeans'], model['Tstds'])), correct_Tms, 1e-2):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Tmeans and Tstds are correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Tmeans, Tstds should be')
        print(correct_Tms)
        print(f'                       Your values are')
        print(model['Tmeans'], model['Tstds'])

except Exception as ex:
    print(f'\n--- 0/{pts} points. model key indexing raised the exception\n')
    print(ex)


######################################################################

print('''\nTesting
  Y = use(X, model)
''')

try:
    pts = 10
    Y = use(X, model)
    correct_Y = np.array([[-0.12808275],
                          [-0.08623466],
                          [-0.04438657],
                          [-0.00253849],
                          [ 0.0393096 ],
                          [ 0.16485386],
                          [ 0.20670194],
                          [ 0.29039811]])

    if np.allclose(Y, correct_Y, 1e-2):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Y should be')
        print(correct_Y)
        print(f'                       Your values are')
        print(Y)

except Exception as ex:
    print(f'\n--- 0/{pts} points. use raised the exception\n')
    print(ex)

######################################################################

print('''\nTesting
  err = rmse(Y, T)
''')

try:
    pts = 10
    err = rmse(Y, T)
    correct_err = 0.0176
    if np.allclose(err, correct_err, 1e-2):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. err should be')
        print(correct_err)
        print(f'                       Your values are')
        print(err)

except Exception as ex:
    print(f'\n--- 0/{pts} points. rmse raised the exception\n')
    print(ex)




name = os.getcwd().split('/')[-1]

print()
print('='*70)
print('{} Execution Grade is {} / 60'.format(name, exec_grade))
print('='*70)

print('''\n __ / 5 Reading in weather.data correctly.''')

print('''\n __ / 5 Count missing values, to show there are some.''')

print('''\n __ / 5 Remove samples with missing values. Count to show there are none.''')

print('''\n __ / 5 Construct X and T matrices as specified.''')

print('''\n __ / 15 Use your train function on X and T. Show results as RMSE values and as plots
       for several different values of learning_rate and n_epochs. Type your 
       observations of RMSE values and of plots with at least five sentences.''')

print('''\n __ / 5 Print the weight values with corresponding variable names. Discuss
       which input variablesa re most signficant in predicting tave values.''')

print()
print('='*70)
print('{} FINAL GRADE is  _  / 100'.format(name))
print('='*70)

print('''
Extra Credit:
Predict the change in tave from one day to the next, instead of the actual tave.
Show and discuss RMSE and plotting results for several values of learning_rate
and n_epochs.''')

print('\n{} EXTRA CREDIT is 0 / 1'.format(name))
