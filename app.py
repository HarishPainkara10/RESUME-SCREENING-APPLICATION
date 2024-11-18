import nltk
import re 
import pickle
import streamlit as st


nltk.download('punkt')
nltk.download('stopwords')  

# loading Clf models and Tfidf model
Clf=pickle.load(open('Clf.pkl','rb'))
Tfidf=pickle.load(open('Tfidf.pkl','rb'))



def CleanResume(Resume_Text):
    # Initialize CleanText with the input text
    CleanText = Resume_Text
    
    # Perform substitutions
    CleanText = re.sub(r'http\S+\s?', ' ', CleanText)  # Remove URLs
    CleanText = re.sub(r'RT|cc', ' ', CleanText)       # Remove retweets and 'cc'
    CleanText = re.sub(r'#\S+\s?', ' ', CleanText)     # Remove hashtags
    CleanText = re.sub('@\S+', '  ', CleanText)  
    CleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', CleanText)
    CleanText = re.sub(r'[^\x00-\x7f]', ' ', CleanText) 
    CleanText = re.sub('\s+', ' ', CleanText)


    return CleanText



# website
def main():
    st.title("Resume Screening Application")
    Uploaded_File = st.file_uploader('Upload Resume',type=['text','pdf'])
   
    if Uploaded_File is not None:
       try:
           Resume_Bytes= Uploaded_File.read()
           Resume_Text= Resume_Bytes.decode('utf-8')
       except UnicodeDecodeError:
           # If UTF-8 Decoding fails,try decoding with 'latin-1'
           Resume_Text= Resume_Bytes.decode('latin-1')
        
       CleanedResume = CleanResume(Resume_Text)
       CleanedResume =Tfidf.transform([CleanedResume])
       Prediction_id =Clf.predict(CleanedResume)[0]
       st.write(Prediction_id)


       # Map category ID to category name
       Category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
       }       
       Category_name =Category_mapping.get(Prediction_id,"Unknown")
       st.write('Predicted Category:-',Category_name)        
    

       
    

   


# python main
if __name__=="__main__":
    main()