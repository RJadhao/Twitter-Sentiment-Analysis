"""Recommendationsystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import index
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls), 
    path('',index.login,name="login"),
    path('logout',index.logout,name="logout"),
    path('myregister', index.register,name="register"),
    path('index', index.index, name="index"),
    path('about',index.about,name="about"),
    path('myprofile',index.myprofile,name="myprofile"),
    path('analysis',index.analysis,name="analysis"),
    path('registration',index.doregister,name="doregister"),
    path('admin',index.admin,name="admin"),
    path('signup',index.dologin,name="dologin"),
    path('prevpred',index.prevpred,name="prevpred"),
    path('analysistext',index.analysistext,name="analysistext"),
    path('splittingsentence',index.splittingsentence,name="splittingsentence"),
    path('tokenization',index.tokenization,name="tokenization"),
    path('stopwordremoval',index.stopwordremoval,name="stopwordremoval"),
    path('stemming',index.stemming,name="stemming"),
    path('viewalluser',index.viewalluser,name="viewalluser"),
    path('adminanalyze',index.adminanalyze,name="adminanalyze"),
    path('viewmypred',index.viewmypred,name="viewmypred"),
    path('viewallpred',index.viewallpred,name="viewallpred"),
    path('prediction_result', index.prediction_result, name='prediction_result'),
    

    
    
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

