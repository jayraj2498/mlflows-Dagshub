### ML-Flow & Dagshub

#### ML flows 
- works in any liabrary language 
- Runs the same way in any cloud 
- designed to scaled from 1 user to large orgnization 
- scaled to big  data with apache spark 

#### ML flow is the open sources platform to manage ML lifecycle :
- it includes 
    - Experimentation 
    - reproducibilliety 
    - deployment 
    - central model registry 

#### ML -flows are provide 4 components 
- ML - Tracking :- here you are able to record all matricx , acccuracy , performance matrix 
- ML flow Project :- we packages all project to run on any platform 
- ML flow model :- deploye ml model in any environment 
- model registry  :- manages model in ccentral repository 



##### run the programe :
(base) PS E:\New folder> python .\app.py
Elasticnet model (alpha=0.500000, l1_ratio=0.500000):
  RMSE: 0.7931640229276851
  MAE: 0.6271946374319587
  R2: 0.10862644997792614  

  - after runnign you will see the foler mlruns :
        inside it all exp will get track : 
        metrics , param all mul value 

  - but we get values wrt to local  


#### ML flows UI 

- in terminal : - mlflow ui  --> u will recieve url 
- check your model performance 



### DagsHub 

- in dags hub acc we can login by using github account 

- add your github repo then you are in dags hub same repo as git have 
- click on remote -> experiments -> = copy these 

export MLFLOW_TRACKING_URI=https://dagshub.com/jayraj2498/mlflows-Dagshub.mlflow \
export MLFLOW_TRACKING_USERNAME=jayraj2498 \
export MLFLOW_TRACKING_PASSWORD=777e2be0b0c43fcc2efbc898716cbaebe35c912b \
python script.py 

- open git bash 
    - run all command 

- next open app.py 

 remote_server_uri="https://dagshub.com/jayraj2498/mlflows-Dagshub.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

we add remoter uri --> now run by gitbash -- python ap.py        
next see the model performance on dif diff parameter 
you can chnage it make new model if require 
you cam send the model to -> production env or staging env 