# Generated by Django 3.0.8 on 2021-04-07 08:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('introml', '0003_auto_20210407_0717'),
    ]

    operations = [
        migrations.AlterField(
            model_name='companyperiodaccountgeneral',
            name='value',
            field=models.CharField(blank=True, default='', max_length=512, null=True),
        ),
    ]
