import torch
import ipfshttpclient
import os
import io
import shutil
import numpy as np
import tensorflow as tf
import json
import h5py
import hashlib




# create an empty list to store all the model hashes
model_hashes =[]

class FSCommunicator:
    def __init__(self, ipfs_path, device):
        self.ipfs_path = ipfs_path
        self.DEVICE = device

        # Connect to the IPFS daemon
        self.client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
        files = self.client.ls(self.ipfs_path)

    def fetch_initial_model(self):
      
        # model_hash = 'QmY1tk29Bwv2eGQV9bKAe7QmCxhBExuYtzn7tXVFUx4tgi'
        # model_bytes = self.client.cat(model_hash)
        # model_path = 'mnist_model_with_weights.h5'
        # with open(model_path, 'wb') as f:
        #     f.write(model_bytes)
        # model = tf.keras.models.load_model(model_path)
        # print("Done Model!!!!!!")
        # opt = "Adam"
        # return model, opt
    
        # model_hash = 'QmY1tk29Bwv2eGQV9bKAe7QmCxhBExuYtzn7tXVFUx4tgi'
        # model_bytes = self.client.cat(model_hash)
        # model_path = 'mnist_model_with_weights.h5'
        # with open(model_path, 'wb') as f:
        #      f.write(model_bytes)
        # model = tf.keras.models.load_model(model_path)
        # #model= modell.get_weights()
        # print("Done Model!!!!!!")
        # print(model) 


        model_hash = 'QmY1tk29Bwv2eGQV9bKAe7QmCxhBExuYtzn7tXVFUx4tgi'
        model_bytes = self.client.cat(model_hash)

        # Load model from bytes as an HDF5 file
        with h5py.File(io.BytesIO(model_bytes), 'r') as f:
            model = tf.keras.models.load_model(f)

        print("Done loading model!")


        # Serialize optimizer state as a JSON string
        opt = model.optimizer
        opt_config = opt.get_config()
        opt_config_converted = {}
        for key, value in opt_config.items():
            if isinstance(value, np.float32):
                value = float(value)
            opt_config_converted[key] = value
        opt_dict = {'name': opt.__class__.__name__, 'config': opt_config_converted}
        opt_str = json.dumps(opt_dict)

        # Serialize optimizer state as a dictionary
        opt_state = {'name': opt.__class__.__name__, 'state_dict': opt.variables()}
        
        return model, opt_state




    def fetch_evaluation_models(self, worker_index, round, num_workers):
        print("Fetch Function")
        state_dicts = []
        
        
        for i in range(num_workers):
            print("worker",i)
            print("hash length",len(model_hashes))
            if i < (len(model_hashes)-1):
                model_hash = model_hashes[i]
                model_bytes = self.client.cat(model_hash)
                with h5py.File(io.BytesIO(model_bytes), 'r') as f:
                    state_dicts.append(tf.keras.models.load_model(f))
                # model_filename = 'model_round_{}_index_{}.h5'.format(round, i)
                # print(model_filename)
                
                # state_dicts.append(tf.keras.models.load_model(f'path/to/model/{model_filename}'))
        # print("Fetch state dicts", state_dicts)
        
      
        return state_dicts  
    


    def push_model(self, state_model, worker_index, round_num, num_workers):
       

        # Clear the model_hashes list if it has reached the number of workers

        if (len(model_hashes)) == num_workers:
                model_hashes.clear()
                print("model_hashes list cleared",model_hashes)

        print("Pushing Model")
        print("Model state",state_model)
        model_filename = 'model_round_{}_index_{}.h5'.format(round_num, worker_index)
        

        state_model.save(f'path/to/model/{ model_filename }')
        # print("state_model",state_model)
        print("MODEL SAVED TO LOCAL")
        

       # Add the file to IPFS and get the new hash
        model_has = self.client.add(f'path/to/model/{model_filename}')
        model_hash = model_has['Hash']
        
        print("Length of model hashes:", len(model_hashes))
        model_hashes.append(model_hash)  # add the new model hash to the list
        print("List of hash:", model_hashes)
        print("Model Hash:", model_hash)
        print("Pushing Complete")

        # Remove the local file
        # os.remove(model_filename)






