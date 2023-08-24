import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
from Model_Building import model_class

app = Flask("__name__")

df_1=pd.read_excel("customer_churn_large_dataset.xlsx")
df_1.drop(['CustomerID',"Name","Churn"],axis=1,inplace=True)

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    

    obj = model_class()
    model = obj.model_build()
    
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6]]
    
    new_df = pd.DataFrame(data, columns = ['Age','Gender', 'Location', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'])
    
    df_2 = pd.concat([df_1, new_df], ignore_index = True) 
   
    
    
    
    
    new_df__dummies = pd.get_dummies(df_2[['Gender','Location']],drop_first=True)
    
    
    final_df=pd.concat([df_2, new_df__dummies], axis=1)
    final_df.drop(['Gender','Location'],axis=1,inplace=True)    
    print(final_df.columns)
    single = model.predict(final_df.tail(1))
    probablity = model.predict_proba(final_df.tail(1))[:,1]
    
    
    if single==1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
        
    return render_template('home.html', output1=o1, output2=o2)
    
app.run()