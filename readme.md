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
