# Generated by Django 4.2.9 on 2024-03-01 06:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0003_rename_uid_sentiment_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='sentiment',
            name='uid',
            field=models.IntegerField(default=0),
        ),
    ]
