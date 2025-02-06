from django.contrib import admin
from application.models import myuser
from application.models import Sentiment

class Adminmyuser(admin.ModelAdmin):
    list_display =('id','username','contact','email','password')

admin.site.register(myuser, Adminmyuser)

class AdminSentiment(admin.ModelAdmin):
    list_display =('uid','text','pred','sentiment_id')

admin.site.register(Sentiment, AdminSentiment)