When you run train.py and specify a folder to save your model to (like: python3 train.py --save_to_folder mytestrun) a folder is created under saved_models
(in this example it will be mytestrun).
For each epoch a subfolder with epoch number is created.
Thus in our example the model trained after 50 epochs will be saved in saved_models/mytestrun/50/, the model trained after 999 epochs will be saved to saved_models/mytestrun/999/.
If you don't see anything in this folder yet, that's OK.
It will start populating after you run training.  
