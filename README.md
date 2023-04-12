

## Code Flow 

 ![alt text](https://raw.githubusercontent.com/bayesianinstitute/FLBLC/main/FLBLC_flow%20diagram.png)




## Setup

### Ganache

1. Go to [Ganache homepage](https://truffleframework.com/ganache) and download.
2. If you are on Linux, you must have received an _.appimage_ file. Follow installation instructions available [here.](https://itsfoss.com/use-appimage-linux/)

### IPFS

- Download and install ipfs in your system link on this to follow this step :

   https://docs.ipfs.tech/install/command-line/#install-official-binary-distributions

### Python

- Download and install Python in your system link on this to follow this step :
[ https://www.python.org/downloads/`](https://www.python.org/downloads/)



## Getting the Application running

### Configuration

#### 1. Ganache

- Open Ganache and click on settings in the top right corner.
- Under **Server** tab:
  - Set Hostname to 127.0.0.1 -lo
  - Set Port Number to 7545
  - Enable Automine
- Under **Accounts & Keys** tab:
  - Enable Autogenerate HD Mnemonic

#### 2. IPFS



- Fire up your terminal and run `ipfs init`
- Then run

  ```
  ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin "[\"*\"]"
  ipfs config --json API.HTTPHeaders.Access-Control-Allow-Credentials "[\"true\"]"
  ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods "[\"PUT\", \"POST\",\"GET\"]"
  ```
-  Run `ipfs daemon` in terimal.  



## Setup
Create a virtual environment and install the requirements.txt by using  below Command.

` python -m venv myenv `
` pip install -r requirements.txt `

###### Smart Contract

1. Install Truffle using `npm install truffle -g`
2. Compile Contracts using `truffle compile`
3. Open new Terminal and deploy contracts using `truffle migrate`





Create a `.env` file containing the private keys of requester and workers in the following format:
```
REQUESTER_KEY=0x...
WORKER1_KEY=0x...
WORKER2_KEY=0x...
WORKER3_KEY=0x...
...
```

Run Application with the following python command 
```
python3 main.py --num_workers 3 --num_rounds 10 
```



