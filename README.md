# MachineLearning_Optimisation
Optimisation program based on DoE and SVR

Here is grouped the optimisation program and files used in the paper. 
The program is described in Program.py with a small procedure at the end.
The two Jupyter Notebooks give the obtained results with the experiments of the different optimisation rounds for the TEMPO and Iodine electrolytes. 
The utilised data are displayed in the different .csv files.

# HOW TO USE THE PROGRAM: exemple of the iodine optimisation

## 1° reading the .csv file, creating and priting the df table
df = pd.read_csv("Iode-1.csv", ';') #make the table as .csv file
##deleting NaNs: if df contains NaN cells when displayed, use the first line below to delete rows with fewer than 'x' non-NaN values (change according to the table obtained) and the 2nd removes columns containing at least one NaN.
#df = df.dropna(thresh=7)
#df= df.dropna(axis='columns')#NaN deletion
colors = plt.cm.tab20(np.linspace(0, 1, 20)[0:len(df.exp.unique())])
color_dic = {label: color for label, color in zip(df.exp.unique(), colors)}
df['color'] = df.exp.map(color_dic) #adds a 'color' column to df to give a unique colour to each experiment
df

## 2° create names
names = ['I2', 'LiI', 'LiTFSI'] #use the exact column names from df here

## 3° Do the ANOVA, here it is done on the PCE
anova(names, 'PCE')

## 4° Optimise the hyperparameters and fit the model
reg_pce=pred_fit('PCE', [0,3]) #here the range chosen in 0-3 as the maximum obtained PCE is 2.71 %

## 5° Plot the surface extrapolation
data_v=[60, 90, 120] #here data_v corresponds to the values of [I2]
surface_plot(reg_pce, 0, 0.6, 0, 0.3, 0, 3, 'viridis') #viridis was chosen as colour for the plotting as it offers a pronounced difference between high and low values. Ocean and magma were also chosen for the AVT and LUEp respectively
