from django.db import models

class myuser(models.Model):
    id =models.AutoField(primary_key=True)
    username = models.CharField(max_length=50)
    contact = models.CharField(max_length=50)
    email = models.CharField(max_length=10)
    password = models.CharField(max_length=13)
    

class Sentiment(models.Model):
    uid = models.IntegerField(default=0, null=False)
    text = models.CharField(max_length=50)
    pred = models.CharField(max_length=50)
    sentiment_id = models.AutoField(primary_key=True)

                    
