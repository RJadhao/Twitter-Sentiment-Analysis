from django.shortcuts import render
import pymysql
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import models
from application.models import myuser
from application.models import Sentiment 
from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
import requests
con=pymysql.connect(host="localhost",user="root",password="root",database="twittersentiment")
def viewallpredexcel(request):
    df=pd.read_csv('C:/Users/dell/Desktop/New folder/tweets.csv')
    for index, row in df.iterrows():
        sentiment_prediction=""
        url = "https://api.meaningcloud.com/sentiment-2.1"
        input_text=row['tweets']
        payload={
        'key': '29cccca1d33d4800f761b3b643214615',
        'txt':input_text,
        'lang': 'en',  # 2-letter code, like en es fr ...
        }

        response = requests.post(url, data=payload)
        json_string = response.text
        try:
            
            data = json.loads(json_string)
            score_tag = data["score_tag"]
            #print("Score Tag:", score_tag)
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", e)
        sentiment_prediction=score_tag
        content={}
        payload=[]
        if(score_tag=="P+" or score_tag=="P"):
            sentiment_prediction="positive"
        if(score_tag=="NEU" or score_tag=="N"):
            sentiment_prediction="negative"
        if(score_tag=="N+" or score_tag=="NONE"):
            sentiment_prediction="neutral"
        print(" Text "+input_text+"Sentiment "+sentiment_prediction)
        content={'text':input_text,'sentiment':sentiment_prediction}
        payload.append(content)
        print("========================")
        content={}
        sentiment = Sentiment(uid='1', text=input_text, pred=sentiment_prediction)

        # Save the Sentiment instance to the database
        sentiment.save()   
    sentiments = Sentiment.objects.all()

    # Count the occurrences of each sentiment (positive, negative, neutral)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for sentiment in sentiments:
        if sentiment.pred == "positive":
            sentiment_counts["positive"] += 1
        elif sentiment.pred == "negative":
            sentiment_counts["negative"] += 1
        elif sentiment.pred == "neutral":
            sentiment_counts["neutral"] += 1

    # Generate the pie chart and get the chart image as a base64 string
    chart_image = generate_pie_chart(sentiment_counts)

    # Prepare data to be passed to the template
    payload = [{'text': sentiment.text, 'pred': sentiment.pred} for sentiment in sentiments]

    return render(request, "adminanalyze.html", {'list': {'items': payload}, 'chart_image': chart_image})

def viewmypred(request):
    content = {}
    payload = []
    uid = request.session['uid']
    q1 = "select * from sentiment"
    #values = (uid,)
    cur = con.cursor()
    cur.execute(q1)
    res = cur.fetchall()
    for x in res:
        content = {'text': x[0], "pred": x[1]}
        payload.append(content)
        content = {}
   
    # Count the occurrences of each sentiment (positive, negative, neutral)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for item in payload:
        sentiment = item["pred"]
        if sentiment == "positive":
            sentiment_counts["positive"] += 1
        elif sentiment == "negative":
            sentiment_counts["negative"] += 1
        elif sentiment == "neutral":
            sentiment_counts["neutral"] += 1

    # Generate the pie chart and get the chart image as a base64 string
    chart_image = generate_pie_chart(sentiment_counts)

    return render(request, "adminanalyze.html", {'list': {'items': payload}, 'chart_image': chart_image})



def index(request):
    return render(request,"index.html")

def admin(request):
    return render(request,"admindashboard.html")

def about(request):
    return render(request,"about.html")

def analysis(request):
    return render(request,"analysis.html")

def adminanalyze(request):
    return render(request,"adminanalyze.html")

def login(request):
    return render(request,"loginpanel.html")
    
def logout(request):
    return render(request,"loginpanel.html")

def register(request):
    return render(request,"registrationPanel.html")


def dologin(request):
    email = request.POST.get('email')
    password = request.POST.get('password')

    try:
        user = myuser.objects.get(email=email, password=password)
    except myuser.DoesNotExist:
        user = None

    if user is not None:
        # Set user information in session
        request.session['id'] = user.id
        request.session['name'] = user.username
        request.session['contact'] = user.contact
        request.session['email'] = user.email
        request.session['password'] = user.password

        return redirect('index')  # Redirect to the index page after successful login
    elif email == "admin" and password == "admin":
        return redirect('adminanalyze') 
    else:
        return render(request, "error.html")




def doregister(request):
    if request.method == 'POST':
        name=request.POST.get('name')
        contact=request.POST.get('contact')
        email=request.POST.get('email')
        password=request.POST.get('password')
        user = myuser(username=name, contact=contact, email=email, password=password)
        user.save()
        return redirect('logout')
            
    return render(request,"registration.html")


def prevpred(request):
    content={}
    payload=[]
    uid=request.session['uid']
    q1="select * from answer where uid=%s";
    values=(uid)
    cur=con.cursor()
    cur.execute(q1,values)
    res=cur.fetchall()
    for x in res:
        content={'answers':x[0]}
        payload.append(content)
        print(payload)
        content={}
    return render(request,"prevpred.html",{'list': {'items':payload}})


def myprofile(request):
    id = request.session.get('id')
    if id is not None:
        name = request.session.get('name')
        contact = request.session.get('contact')
        email = request.session.get('email')
        
        payload = [{
            'name': name,
            'contact': contact,
            'email': email
        }]
        return render(request, "viewprofile.html", {'list': {'items': payload}})
    else:
        messages.error(request, "User not logged in.")
        return redirect('login')  




def splittingsentence(request):
    input_text = request.POST.get('sentence')
    request.session['input_text'] = input_text
    from nltk import sent_tokenize
    sentences = sent_tokenize(input_text)
    return render(request,"splittingsentence.html",{'sentences':sentences , 'input_text':input_text})

def tokenization(request):
    input_text = request.POST.get('sentencs')
    print(input_text)
    input_text = request.session.get('input_text')
    from nltk import sent_tokenize, word_tokenize
    sentences = sent_tokenize(input_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    return render(request, "tokenization.html", {'tokens': tokenized_sentences, 'input_text': input_text})

def stopwordremoval(request):
    from nltk.corpus import stopwords
    from nltk import sent_tokenize, word_tokenize
    input_text = request.session.get('input_text')
    sentences = sent_tokenize(input_text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    stop_words = set(stopwords.words('english'))
    without_stopwords = [[word for word in tokens if word.lower() not in stop_words] for tokens in tokenized_sentences]

    return render(request, "stopwordremoval.html", {'stopword':  without_stopwords, 'input_text': input_text})


def stemming(request):  
    import nltk
    input_text = request.session.get('input_text')
    snow_stemmer = nltk.SnowballStemmer("english")
    stemming = ' '.join(snow_stemmer.stem(i) for i in input_text.split(' '))

 
    return render(request, "stemming.html", {'stemming':  stemming, 'input_text': input_text})

    # Function to calculate sentiment prediction for a single input text


def get_sentiment_prediction(text):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    # Instantiate the BERT tokenizer and model for sequence classification
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
    # Perform sentiment prediction using the model
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Get the predicted sentiment probabilities
        sentiment_probabilities = torch.softmax(outputs.logits, dim=1).tolist()[0]
        
        # Get the predicted sentiment label (0, 1, or 2) from the probabilities
        sentiment_label = sentiment_probabilities.index(max(sentiment_probabilities))
        
        # Convert the sentiment label to its corresponding text representation
        sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = sentiment_mapping.get(sentiment_label, "Unknown")
        
    return sentiment

def prediction_result(request):
    # Get the sentiment_prediction from the session
    sentiment_prediction = request.session.get('sentiment_prediction', None)

    context = {
        'sentiment_prediction': sentiment_prediction
    }

    return render(request, 'finalprediction.html', context)

def analysistext(request):
    if request.method == 'POST':
        input_text = request.session.get('input_text')

        # Get the sentiment prediction for the input text
        sentiment_prediction = get_sentiment_prediction(input_text)

        # Get the logged-in user's ID
        user_id = request.session.get('id')

        try:
            # Retrieve the MyUser object based on the user_id
            user = myuser.objects.get(id=user_id)
        except myuser.DoesNotExist:
            return HttpResponse("User with the provided ID does not exist.")

        # Create a Sentiment instance and set the 'uid', 'text', and 'pred' fields
        sentiment = Sentiment(uid=user.id, text=input_text, pred=sentiment_prediction)

        # Save the Sentiment instance to the database
        sentiment.save()

        # Set the sentiment_prediction in the session
        request.session['sentiment_prediction'] = sentiment_prediction

        # Redirect to the 'prediction_result' view
        return redirect('prediction_result')

    return render(request, "finalprediction.html", {'sentiment_prediction': sentiment_prediction})
    
    #return render(request,"finalprediction.html",{'sentiment_prediction':sentiment_prediction})

#return redirect('http://127.0.0.1:8000/')


def viewalluser(request):
    users = myuser.objects.all()
    payload = []

    for user in users:
        payload.append({'name': user.username, 'contact': user.contact, 'email': user.email})

    return render(request, "viewalluser.html", {'list': {'items': payload}})


import matplotlib.pyplot as plt
import io
import base64

def generate_pie_chart(sentiment_counts):
    labels = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    colors = ["green", "red", "gray"]  # You can customize the colors here
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title("Sentiment Analysis")
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

    # Save the chart as a bytes object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode the chart image as a base64 string
    chart_image = base64.b64encode(buffer.read()).decode('utf-8')

    return chart_image

import json
def viewmypred(request):
    # Check if 'id' is set in the session
    if 'id' not in request.session:
        return HttpResponse("ID is not set in session. Please set the ID before accessing this page.")

    # If 'id' is set in the session, retrieve it
    user_id = request.session['id']
    
    # Retrieve all sentiments where the uid matches the user's ID
    sentiments = Sentiment.objects.filter(uid=user_id)
    
    # Count the occurrences of each sentiment (positive, negative, neutral)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for sentiment in sentiments:
        sentiment_counts[sentiment.pred] += 1
    
    # Prepare data to be passed to the template
    chart_data = {
        "labels": list(sentiment_counts.keys()),
        "data": list(sentiment_counts.values())
    }
    
    # Convert chart_data to JSON format
    chart_data_json = json.dumps(chart_data)
    
    return render(request, "viewmypred.html", {'sentiments': sentiments, 'chart_data_json': chart_data_json})

def uploadfileanalyze(request):
    import pandas as pd
    file =request.POST.get('file')
    df = pd.read_csv(file)

    return render(request,adminanalyze.html)

def viewallpred(request):
    # Retrieve all sentiment objects from the database
    sentiments = Sentiment.objects.all()

    # Count the occurrences of each sentiment (positive, negative, neutral)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for sentiment in sentiments:
        if sentiment.pred == "positive":
            sentiment_counts["positive"] += 1
        elif sentiment.pred == "negative":
            sentiment_counts["negative"] += 1
        elif sentiment.pred == "neutral":
            sentiment_counts["neutral"] += 1

    # Generate the pie chart and get the chart image as a base64 string
    chart_image = generate_pie_chart(sentiment_counts)

    # Prepare data to be passed to the template
    payload = [{'text': sentiment.text, 'pred': sentiment.pred} for sentiment in sentiments]

    return render(request, "adminanalyze.html", {'list': {'items': payload}, 'chart_image': chart_image})
