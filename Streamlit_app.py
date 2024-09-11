import streamlit as st
from streamlit_extras.row import row
from streamlit_extras.grid import grid 

import pandas as pd
import numpy as np 
from chembl_webresource_client.new_client import new_client
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import pickle
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.feat import CircularFingerprint
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher
import gzip


def smiles_to_image(smiles,id):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(400, 400),caption=f"Potential Ligand {id}")
    else:
        return None



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
        font-weight: 600;
        color:#333333;
    }
    .st-emotion-cache-jkfxgf.e1nzilvr5 p{
        font-size: 30px;
        font-weight: 600;
        color:#1D4E89;
            
            
            }
    .st-emotion-cache-1h9usn1.eqpbllx3 {
        background-color:#F5F5F5;  /*  #001f3f; */
        color: white;  /* Font color */
        padding: 10px;
        border-radius: 10px;
    }
    st-emotion-cache-1clstc5 eqpbllx0{
        background-color: #FFFFFF;  /* Soft White Background for content */
        color: #333333;  /* Charcoal Font Color */
            }
    p {
        font-size: 20px;
        font-weight: 400;
        color:#333333;
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

        font-weight: 600;
        color:#333333;
    }
    
    /* You can also target specific tags */
    h2, h3 {
        font-weight: 600;
    }
    
    
    /* Customize more as needed */
    </style>
    """, unsafe_allow_html=True)

# Streamlit content
with open('featurizer.pickle', 'rb') as file:
    loaded_featurizer = pickle.load(file)
with open('random_forest_class.pickle', 'rb') as file:
    random_forest_model = pickle.load(file)

# all_candidates=pd.read_pickle('ALL_Candidates.pkl')

with gzip.open('./chembl_30_chemreps_tenth.txt.gz', "rt") as file:
    # Assuming the file is tab-separated and has headers
    all_candidates = pd.read_csv(file, sep='\t', usecols=['chembl_id', 'canonical_smiles','standard_inchi','standard_inchi_key'])



st.title("Drug Discovery for Dengue Fever")
with st.expander('Ligand/Target interaction Prediction',expanded=True):
    try:
        st.write('')
        st.write('')
        st.write('Draw your the structure of your proposed ligand and press apply to predict if it is an active ligand:')
        smiles = st_ketcher()
        print(smiles)
        row_to_check=loaded_featurizer(smiles).tolist()
        print(row_to_check)
        y_output=random_forest_model.predict(row_to_check)
        if y_output[0]==1:
            # with col1:
            #     st.write(row['chembl_id'])
            # with col2:
            st.write('This ligand has intermediate interaction with the target protein') 
        else:           
            st.write('This ligand has no interaction with your target protein')
    except:
        pass

with st.expander('Drug Discovery Dashboard'):
    
    counter_drugs=0


    my_grid = grid(1,1,2, 2, 2, 2,2,2,2, vertical_align="center")
    my_grid.write('Using the model explained in the expander above, we can detect if there is any potential ligand which has interaction with the associated target protein. We use the ChemBL database which has 2136187 compounds. If you want to find new ligands press the button below:')
    button_discover=my_grid.button('Discover Ligands')
    if button_discover:
        my_grid.write('Ligand ID')
        my_grid.write('Ligand Structure')
        for index,row in all_candidates.iloc[1:10000,:].iterrows():
            try:
                row_to_check=loaded_featurizer(row['canonical_smiles']).tolist()
                y_output=random_forest_model.predict(row_to_check)
                if y_output[0]==1:
                    # with col1:
                    #     st.write(row['chembl_id'])
                    # with col2:
                        
                    image = smiles_to_image(row['canonical_smiles'],row['chembl_id'])
                    if image:
                        my_grid.write(row['chembl_id'])
                        my_grid.image(image)
                    counter_drugs+=1
                    if counter_drugs==4:
                        break
                        

            except Exception as er:
                #  st.warning(er)
                # st.write(row['canonical_smiles'])
                # st.write(er)
                pass



with st.expander('Machine Learning Model Development Steps Summary'):
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
    st.write("There are some important points needed to be noted. It seams that there is no Active ligands for our target. Therefore, the process of drug discovery will be limited to find intermedaited compounds. Also, since the dataset is imbalanced, we need to use a technique to balance our database for the sake of finding the intermdiate ligands more accurately.")

    # col1,col2=st.columns((1,1))



