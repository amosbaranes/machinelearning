# Generated by Django 3.0.8 on 2020-08-15 14:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_auto_20200814_2159'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='price',
            options={'ordering': ('-price_date',), 'verbose_name': 'price', 'verbose_name_plural': 'prices'},
        ),
    ]
