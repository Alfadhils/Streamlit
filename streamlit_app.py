import streamlit as st
import pandas as pd

# Import for API calls
import requests

# Import for navbar
from streamlit_option_menu import option_menu

# Import for dynamic tagging
from streamlit_tags import st_tags, st_tags_sidebar

# Imports for aggrid
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode

# Import for loading interactive keyboard shortcuts into the app
# from dashboard_utils.gui import keyboard_to_url
# from dashboard_utils.gui import load_keyboard_class

if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"

st.set_page_config(layout=layout, page_title="Zero-Shot Text Classifier", page_icon="🤗")

# Set up session state so app interactions don't reset the app.
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

#######################################################

# The block of code below is to display the title, logos and introduce the app.

c1, c2 = st.columns([0.4, 2])

with c1:
    st.image(
        "logo.png",
        width=110,
    )

with c2:
    st.caption("")
    st.title("Zero-Shot Text Classifier")


st.sidebar.image(
    "30days_logo.png",
)

st.write("")

st.markdown(
    """

Classify keyphrases fast and on-the-fly with this mighty app. No ML training needed!

Create classifying labels (e.g. `Positive`, `Negative` and `Neutral`), paste your keyphrases, and you're off!  

"""
)

st.write("")

st.sidebar.write("")

with st.sidebar:
    selected = option_menu(
        "",
        ["Demo", "Unlocked Mode"],
        icons=["bi-joystick", "bi-key-fill"],
        menu_icon="",
        default_index=0,
    )
    
with st.sidebar.expander('General Workflow'):
  st.markdown('This app receives input the keyphrases and labels of text classification and uses HuggingFace API to predict the output. The model used is the distilbart-mnli.')
  
st.sidebar.markdown("---")

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    """

App created by [Alfadhils](https://github.com/Alfadhils) for the 30 days [Streamlit](https://streamlit.io/).
The app is heavily based on the [tutorial](https://www.charlywargnier.com/post/how-to-create-a-zero-shot-learning-text-classifier-using-hugging-face-and-streamlit) 
and source [code](https://github.com/CharlyWargnier/zero-shot-classifier/) of original app by Charlie Wargnier.  

The app provide three different models to select, the models are :
1. [distilbart-mnli-12-3](https://huggingface.co/valhalla/distilbart-mnli-12-3)
2. [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
3. [DeBERTa-v3-mnli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)

"""
)

demo = True if selected == 'Demo' else False

with st.form(key="my_form"):
    if demo :
        API_KEY = st.secrets["API_TOKEN"]
    else :
        API_KEY = st.text_input(
            "Enter your 🤗 HuggingFace API key",
            help="Once you created you HuggingFace account, you can get your free API token in your settings page: https://huggingface.co/settings/tokens",
            value="YOUR-HF-API-KEY"
        )

    url_map = {
        "distil-bart" : "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3",
        "bart-large" :  "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
        "DeBERTa-V3" : "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    }
    
    model = st.selectbox(
        'Select your preffered model:',
        ('distil-bart', 'bart-large', 'DeBERTa-v3'),
        help="The details of the selected model can be accessed through the About section on the sidebar"
    )
    
    API_URL = url_map[model]

    headers = {"Authorization": f"Bearer {API_KEY}"}

    label_widget = st_tags(
                label="Enter desired classification labels",
                text="Add labels - 3 max",
                value=["Transactional", "Informational"],
                suggestions=[
                    "Navigational",
                    "Transactional",
                    "Informational",
                    "Positive",
                    "Negative",
                    "Neutral",
                ],
                maxtags=3,
            )

    new_line = "\n"
    nums = [
        "I want to buy something in this store",
        "How to ask a question about a product",
        "Request a refund through the Google Play store",
        "I have a broken screen, what should I do?",
        "Can I have the link to the product?",
    ]

    sample = f"{new_line.join(map(str, nums))}"

    MAX_LINES = 5 if demo else 50
    text = st.text_area(
                "Enter keyphrase to classify",
                sample,
                height=200,
                key="2",
                help="At least two keyphrases for the classifier to work, one per line, "
                + str(MAX_LINES)
                + " keyphrases max as part of the demo",
            )
    lines = text.split("\n")  # A list of lines
    linesList = []
    for x in lines:
        linesList.append(x)
    linesList = list(dict.fromkeys(linesList))
    linesList = list(filter(None, linesList))

    if len(linesList) > MAX_LINES:
        st.info(
            f"🚨 Only the first "
            + str(MAX_LINES)
            + (" keyprases will be reviewed. Unlock that limit by switching to 'Unlocked Mode'" if demo 
               else " keyphrases will be reviewed. The limitation preserves performance for inference")
        )
    
    linesList = linesList[:MAX_LINES]

    submit_button = st.form_submit_button(label="Submit")

if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

elif submit_button and not text:
    st.warning("❄️ There is no keyphrases to classify")
    st.session_state.valid_inputs_received = False
    st.stop()

elif submit_button and not label_widget:
    st.warning("❄️ You have not added any labels, please add some! ")
    st.session_state.valid_inputs_received = False
    st.stop()

elif submit_button and len(label_widget) == 1:
    st.warning("❄️ Please make sure to add at least two labels for classification")
    st.session_state.valid_inputs_received = False
    st.stop()

elif not API_KEY :
    st.warning("❄️ Please make sure to add at your personal API Key for the 'unlocked mode' or use 'demo mode'")
    st.session_state.valid_inputs_received = False
    st.stop()

elif submit_button or st.session_state.valid_inputs_received:
    if submit_button:
        st.session_state.valid_inputs_received = True

    listToAppend = []
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        # st.write('request status code : ', response.status_code)
        return response.json()

    for row in linesList:
        output2 = query(
                    {
                        "inputs": row,
                        "parameters": {"candidate_labels": label_widget},
                        "options": {"wait_for_model": True},
                    }
                )
        listToAppend.append(output2)

    df = pd.DataFrame.from_dict(listToAppend)
    if df.empty or 'error' in df.columns:
        st.warning("❄️ No response succesful, please review your input and try again")
        st.session_state.valid_inputs_received = False
        st.stop()
        
    st.success("✅ Success!")
    
    st.caption("")
    st.markdown("### Check classifier results")
    st.caption("")
    
    st.checkbox(
                "Widen layout",
                key="widen",
                help="Tick this box to toggle the layout to 'Wide' mode",
            )
    
    f = [[f"{x:.2%}" for x in row] for row in df["scores"]]
    df["classification scores"] = f
    df.drop("scores", inplace=True, axis=1)
    df.rename(columns={"sequence": "keyphrase"}, inplace=True)

    gb = GridOptionsBuilder.from_dataframe(df)
    # enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
    gb.configure_default_column(
        enablePivot=True, enableValue=True, enableRowGroup=True
    )
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
    gridOptions = gb.build()

    response = AgGrid(
        df,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        height=400,
        fit_columns_on_grid_load=False,
        configure_side_bar=True,
    )

    cs, c1 = st.columns([2, 2])

    with cs:

        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode("utf-8")

        csv = convert_df(df)  #

        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv",
        )