import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

FEATURES=["no_of_dependents","education","self_employed","income_annum","loan_amount",
"loan_term_years","cibil_score","residential_assets_value","commercial_assets_value",
"luxury_assets_value","bank_balance"]
SCALE=["income_annum","loan_amount","cibil_score","residential_assets_value",
"commercial_assets_value","luxury_assets_value","bank_balance"]

def clean(df):
    df=df.copy()
    df.columns=[str(c).strip() for c in df.columns]
    return df.rename(columns={"loan_term":"loan_term_years","bank_asset_value":"bank_balance"})

def prep(df):
    df=clean(df)
    if df["education"].dtype=="object": df["education"]=df["education"].str.strip().map({"Graduate":0,"Not Graduate":1})
    if df["self_employed"].dtype=="object": df["self_employed"]=df["self_employed"].str.strip().map({"No":0,"Yes":1})
    if df["loan_status"].dtype=="object": df["loan_status"]=df["loan_status"].str.strip().map({"Approved":0,"Rejected":1})
    return df

@st.cache_data
def load_uploaded(file):
    return pd.read_csv(file)

def train(df):
    df=prep(df)
    data=df[FEATURES+["loan_status"]].dropna()
    X=data[FEATURES].copy(); y=data.loan_status.astype(int)
    scaler=StandardScaler(); X[SCALE]=scaler.fit_transform(X[SCALE])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    model=XGBClassifier(learning_rate=.1,max_depth=5,n_estimators=300,random_state=42,eval_metric="logloss")
    model.fit(Xtr,ytr)
    return model,scaler,Xte,yte,model.predict(Xte)

st.title("🏦 Loan Approval Prediction")
st.caption("Complete Data Science & Machine Learning Project")

section=st.sidebar.radio("📌 Project Sections",[
"Overview","Dataset","EDA","Preprocessing","Model Comparison","Best Model",
"Evaluation","Feature Importance","Loan Prediction","Conclusion"])

uploaded=st.sidebar.file_uploader("Upload loan CSV",type=["csv"])
df=None
if uploaded: df=load_uploaded(uploaded)
else:
    from pathlib import Path
    files=list(Path("data").glob("*.csv")) if Path("data").exists() else []
    if files: df=pd.read_csv(files[0])
if df is not None: df=clean(df)

if section=="Overview":
    st.header("1. Project Overview")
    st.subheader("Business Problem")
    st.write("Financial institutions receive many loan applications. Manual evaluation can be time-consuming and may lead to inconsistent decisions. The project builds a Machine Learning model to predict whether a loan application should be approved or rejected.")
    a,b,c=st.columns(3); a.metric("ML Type","Supervised"); b.metric("Problem","Classification"); c.metric("Target","loan_status")
    st.subheader("Workflow")
    st.info("Business Problem → Data Acquisition → Data Preparation → EDA → Modeling → Evaluation → Prediction → Deployment")
    st.subheader("Target Variable")
    st.write("loan_status: Approved / Rejected")

elif section=="Dataset":
    st.header("2. Dataset Information")
    if df is None: st.warning("Upload your loan CSV using the sidebar.")
    else:
        a,b,c=st.columns(3); a.metric("Rows",f"{len(df):,}"); b.metric("Columns",len(df.columns)); c.metric("Missing Values",int(df.isna().sum().sum()))
        st.subheader("Dataset Preview"); st.dataframe(df.head(10),use_container_width=True)
        st.subheader("Column Information")
        info=pd.DataFrame({"Column":df.columns,"Data Type":[str(df[x].dtype) for x in df.columns],"Non-Null":[df[x].notna().sum() for x in df.columns]})
        st.dataframe(info,use_container_width=True)
        desc={"loan_id":"Unique loan application ID","no_of_dependents":"Number of dependents","education":"Applicant education level","self_employed":"Employment status","income_annum":"Annual income","loan_amount":"Requested loan amount","loan_term":"Loan term in years","cibil_score":"Credit score","residential_assets_value":"Residential assets value","commercial_assets_value":"Commercial assets value","luxury_assets_value":"Luxury assets value","bank_asset_value":"Bank balance","loan_status":"Final loan decision"}
        st.table(pd.DataFrame({"Feature":list(desc),"Description":list(desc.values())}))

elif section=="EDA":
    st.header("3. Exploratory Data Analysis")
    if df is None: st.warning("Upload the CSV to display live EDA.")
    else:
        if "loan_status" in df:
            st.subheader("Loan Status Distribution")
            fig,ax=plt.subplots(); df.loan_status.value_counts().plot(kind="bar",ax=ax); ax.set_xlabel("Loan Status"); ax.set_ylabel("Count"); st.pyplot(fig)
        st.subheader("Feature Distribution")
        cols=df.select_dtypes(include=np.number).columns.tolist()
        col=st.selectbox("Choose feature",cols)
        fig,ax=plt.subplots(); df[col].dropna().plot(kind="hist",bins=30,ax=ax); ax.set_title(col); st.pyplot(fig)
        st.subheader("Summary Statistics"); st.dataframe(df.describe().T,use_container_width=True)

elif section=="Preprocessing":
    st.header("4. Data Preprocessing")
    st.markdown("**Steps followed:**")
    st.write("1. Load dataset  2. Identify target variable  3. Encode categorical variables  4. Separate features and target  5. Apply StandardScaler  6. Train/test split (80/20, random_state=42)")
    st.code("education: Graduate=0, Not Graduate=1\nself_employed: No=0, Yes=1\nloan_status: Approved=0, Rejected=1")
    st.write("Scaled features:",", ".join(SCALE))

elif section=="Model Comparison":
    st.header("5. Model Comparison")
    r=pd.DataFrame({"Model":["Logistic Regression","Decision Tree","Random Forest","KNN","SVM","AdaBoost","XGBoost"],"Accuracy":[.900468,.967213,.976581,.596019,.627635,.969555,.984778]})
    r["Accuracy (%)"]=r.Accuracy*100; st.dataframe(r[["Model","Accuracy (%)"]].style.format({"Accuracy (%)":"{:.2f}%"}),use_container_width=True)
    fig,ax=plt.subplots(); ax.bar(r.Model,r["Accuracy (%)"]); ax.set_ylim(0,100); ax.set_ylabel("Accuracy (%)"); ax.tick_params(axis="x",rotation=30); st.pyplot(fig)
    st.success("XGBoost is the best-performing model with 98.48% test accuracy.")

elif section=="Best Model":
    st.header("6. Best Model — XGBoost")
    st.write("XGBoost achieved the highest accuracy among the tested models.")
    st.subheader("GridSearchCV Best Parameters")
    st.table(pd.DataFrame({"Parameter":["learning_rate","max_depth","n_estimators"],"Best Value":[0.1,5,300]}))
    st.code("XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=300)")
    st.metric("Test Accuracy","98.48%")

elif section=="Evaluation":
    st.header("7. Model Evaluation")
    st.metric("Test Accuracy","98.48%")
    cm=np.array([[534,2],[11,307]])
    st.subheader("Confusion Matrix")
    fig,ax=plt.subplots(); ax.imshow(cm); ax.set_xticks([0,1],["Approved","Rejected"]); ax.set_yticks([0,1],["Approved","Rejected"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    st.pyplot(fig)
    st.write("Recorded confusion matrix: [[534, 2], [11, 307]].")

elif section=="Feature Importance":
    st.header("8. Feature Importance")
    imp=pd.DataFrame({"Feature":["cibil_score","loan_term_years","loan_amount","income_annum","education","no_of_dependents","luxury_assets_value","residential_assets_value","self_employed","loan_id","commercial_assets_value","bank_balance"],"Importance":[.754444,.168715,.016074,.015750,.007337,.006892,.006574,.006022,.005192,.005003,.004911,.0031]})
    st.dataframe(imp.sort_values("Importance",ascending=False),use_container_width=True)
    fig,ax=plt.subplots(); q=imp.sort_values("Importance"); ax.barh(q.Feature,q.Importance*100); ax.set_xlabel("Importance (%)"); st.pyplot(fig)
    st.success("CIBIL score and loan term together account for more than 92% of feature importance in the notebook analysis.")

elif section=="Loan Prediction":
    st.header("9. Interactive Loan Prediction")
    if df is None: st.warning("Upload your CSV first so the app can train the model.")
    else:
        try:
            model,scaler,_,_,_=train(df)
            with st.form("loan"):
                a,b,c=st.columns(3)
                dep=a.number_input("No. of Dependents",0,20,2)
                edu=b.selectbox("Education",["Graduate","Not Graduate"])
                emp=c.selectbox("Self Employed",["No","Yes"])
                income=a.number_input("Annual Income",0,100000000,5000000,step=100000)
                amount=b.number_input("Loan Amount",0,100000000,10000000,step=100000)
                term=c.number_input("Loan Term (Years)",1,50,10)
                cibil=a.number_input("CIBIL Score",300,900,700)
                res=b.number_input("Residential Assets Value",0,100000000,10000000,step=100000)
                com=c.number_input("Commercial Assets Value",0,100000000,5000000,step=100000)
                lux=a.number_input("Luxury Assets Value",0,100000000,5000000,step=100000)
                bank=b.number_input("Bank Balance",0,100000000,1000000,step=100000)
                submit=st.form_submit_button("🔮 Predict Loan Status")
            if submit:
                row=pd.DataFrame([{"no_of_dependents":dep,"education":0 if edu=="Graduate" else 1,"self_employed":0 if emp=="No" else 1,"income_annum":income,"loan_amount":amount,"loan_term_years":term,"cibil_score":cibil,"residential_assets_value":res,"commercial_assets_value":com,"luxury_assets_value":lux,"bank_balance":bank}])
                row[SCALE]=scaler.transform(row[SCALE])
                raw_pred=model.predict(row[FEATURES])[0]
                if isinstance(raw_pred, str):
                    label=raw_pred.strip().lower()
                    p=0 if label=="approved" else 1 if label=="rejected" else int(raw_pred)
                else:
                    p=int(raw_pred)
                classes=list(model.classes_)
                pred_index=classes.index(raw_pred)
                prob=model.predict_proba(row[FEATURES])[0][pred_index]*100
                (st.success if p==0 else st.error)(f"{'✅ Approved' if p==0 else '❌ Rejected'}")
                st.metric("Model Confidence",f"{prob:.2f}%")
        except Exception as e: st.error(str(e))

else:
    st.header("10. Project Insights & Conclusion")
    st.markdown("""
    **Key Insights**
    - This is a supervised binary classification problem.
    - XGBoost performed best among the tested models.
    - Test accuracy is 98.48%.
    - CIBIL score is the strongest feature.
    - Loan term is the second major contributor.
    - Asset-related features have comparatively smaller importance.

    **Business Value**
    The model can support faster and more consistent loan screening and data-driven decision support. It should be validated for financial, legal and fairness requirements before real-world use.
    """)
    st.success("End-to-end project completed: Understanding → EDA → Preprocessing → Modeling → Tuning → Evaluation → Deployment.")
