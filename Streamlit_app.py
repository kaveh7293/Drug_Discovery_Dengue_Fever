import streamlit as st
import pandas as pd
import numpy as np 
from chembl_webresource_client.new_client import new_client
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer

target = new_client.target
target_query = target.search('dengue fever')
targets = pd.DataFrame.from_dict(target_query)
selected_target = targets.target_chembl_id[12]
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
df = pd.DataFrame.from_dict(res)
df2 = df[df.standard_value.notna()]

bioactivity_class = []
for i in df2.standard_value:
  if float(i) >= 10000:
    bioactivity_class.append("inactive")
  elif float(i) <= 1000:
    bioactivity_class.append("active")
  else:
    bioactivity_class.append("intermediate")

ECFP_features=pd.read_pickle('ECFP_Classification_features.pkl')

st.markdown("""
    <style>
    /* Make all text in the app bold */
    body {
        font-size: 15px;
        font-weight: bold;
    }
    h1{
        font-size: 35px;
        font-weight: bold;
    }
    h2{
        font-size: 20px;

        font-weight: bold;
    }
    li{
        font-size: 15px;

        font-weight: bold;
    }
    
    /* You can also target specific tags */
    h2, h3, p {
        font-weight: bold;
    }
    
    
    /* Customize more as needed */
    </style>
    """, unsafe_allow_html=True)

# Streamlit content
st.title("Drug Discovery for Dengue Fever")
st.subheader("Steps followed to find the drug candidates:")
st.markdown("""
    <ul><li>Find the protein target from the ChEMBL Database. The following protein is used as a target:<br>CHEMBL5980 (Dengue virus type 2 NS3 protein)..<br> </li>
        <li>Find the ligands associated with the NS3 protein and filter them based on the availability of their IC50. </li>.<br></ul>

    """, unsafe_allow_html=True)


st.write(df[['activity_id','assay_chembl_id','canonical_smiles','standard_units','standard_value']])
st.markdown("""
    <ul><li>Assigin activity levels to each of the ligands based on their IC50. The associated activity levels are called <em>Active</em> (i.e., IC50 less than 1000 nM), <em>Intermediate</em> (i.e., IC50 between 1000 to 10000 nM) and <em>Inactive</em> (i.e., IC50 greater than 10000 nM).<br></li>
        <li>In the next step a featurizer <em> Extended Connectivity Fingerprints (ECFP featurizer)</em> is used to create features using the SMILES in the above database. In the following you can see the resulting dataframe that will be used to train the associated models.   </li></ul>

    """, unsafe_allow_html=True)
st.write(ECFP_features)
class_report_df=pd.read_pickle('class_report.pkl')
st.markdown("""
    <ul><li>After 80/20 split of training and testing datasets, the following metrics were obtained for the trained RandomForestClassifier.<br></li>
          </ul>

    """, unsafe_allow_html=True)
st.write(class_report_df)

