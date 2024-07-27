in this model, I will explain the actual MorpheusML model. The model could be called from the terminal, see here: https://morpheus.gitlab.io/faq/general/software-structure/. Using the terminal, one can specify the model and other parameters, however, the parameters can not be passed. To do so, one needs to use the FitMulticell package, see here for a simple example: https://fitmulticell.readthedocs.io/en/latest/example/minimal.html#Inference-problem-definition. Within the Python script, one can call the model and pass the parameter vector.

In the Cell movement model, we currently fit 4 different parameters, see the "cell_migration.py" file

I wrote some comments on this file: "synth_data_params_4_wass_var_50cell/cell_migration.py" to explain the file's main components
