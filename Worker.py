import json
from web3 import Web3, HTTPProvider

import torch.optim as optim
from BCCommunicator import BCCommunicator
from FSCommunicator import FSCommunicator
from Model import Model
import tensorflow as tf


class Worker:
    truffle_file = json.load(open('./build/contracts/FLTask.json'))


    
    def __init__(self, ipfs_path, device, num_workers, idx, topk, key, is_evil):
        self.bcc = BCCommunicator()
        self.fsc = FSCommunicator(ipfs_path, device)
        
        model, opt_state = self.fsc.fetch_initial_model()
        self.is_evil = is_evil
        
        # Load optimizer
        class_ = getattr(tf.optimizers, opt_state['name'])
        opt = class_()
        opt.build(model.trainable_variables)
        opt_weights = [tf.convert_to_tensor(w) for w in opt_state['state_dict']]
        opt.set_weights(opt_weights)



        self.model = Model(num_workers, idx, model, opt, device, topk, is_evil)
        self.idx = idx
        self.num_workers = num_workers

        self.key = key
        self.w3 = Web3(HTTPProvider("http://localhost:7545"))
        if self.w3.isConnected():
            print("Worker initialization: connected to blockchain")

        self.account = self.w3.eth.account.privateKeyToAccount(key)
        self.contract = self.w3.eth.contract(bytecode=self.truffle_file['bytecode'], abi=self.truffle_file['abi'])    
    
    
    def train(self, round):
        print("TRAINING")
        cur_state_dict = self.model.train()
        # print("cur_state_dict:",cur_state_dict)
        
        self.fsc.push_model(cur_state_dict, self.idx, round, self.num_workers)  
    
    
    def evaluate(self, round):
        print("Evaluating")
        # retrieve all models of the other workers
        state_dicts = self.fsc.fetch_evaluation_models(self.idx, round, self.num_workers)
            
        ranks, topk_dicts, unsorted_scores = self.model.eval(state_dicts)
        
        # add our own model for the averaging
        topk_dicts.append(self.model.model)
       
        # TODO add blockchain functionality with sending the ranks to BC here
            
        return self.model.average(topk_dicts), topk_dicts, unsorted_scores 

        
    
    
    def update_model(self, avg_dicts):
        # here we update the model with the averaged dicts
        self.model.adapt_current_model(avg_dicts)

    def join_task(self, contract_address):
        self.contract_address = contract_address
        self.contract_instance = self.w3.eth.contract(abi=self.truffle_file['abi'], address=contract_address)

        tx = self.contract_instance.functions.joinTask().buildTransaction({
            "gasPrice": self.w3.eth.gas_price, 
            "chainId": 1337, 
            "from": self.account.address, 
            'nonce': self.w3.eth.getTransactionCount(self.account.address)
        })
        #Get tx receipt to get contract address
        signed_tx = self.w3.eth.account.signTransaction(tx, self.key)
        tx_hash = self.w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.getTransactionReceipt(tx_hash)

    def get_model_uri(self):
        return self.contract_instance.functions.getModelURI().call()

    def get_round_number(self):
        return self.contract_instance.functions.getRound().call()

       
