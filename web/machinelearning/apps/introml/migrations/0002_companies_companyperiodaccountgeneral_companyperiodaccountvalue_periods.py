# Generated by Django 3.0.8 on 2021-04-07 03:53

from django.db import migrations, models
import django.db.models.deletion
import machinelearning.apps.introml.models


class Migration(migrations.Migration):

    dependencies = [
        ('introml', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Companies',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cik', models.CharField(blank=True, default='', max_length=20, null=True)),
            ],
            options={
                'verbose_name': 'company',
                'verbose_name_plural': 'companies',
                'ordering': ['cik'],
            },
            bases=(models.Model, machinelearning.apps.introml.models.ModifyModel),
        ),
        migrations.CreateModel(
            name='Periods',
            fields=[
                ('year_quarter', models.IntegerField(primary_key=True, serialize=False)),
                ('year', models.SmallIntegerField(default=0)),
                ('quarter', models.SmallIntegerField(default=0)),
            ],
            options={
                'verbose_name': 'year',
                'verbose_name_plural': 'years',
            },
            bases=(models.Model, machinelearning.apps.introml.models.ModifyModel),
        ),
        migrations.CreateModel(
            name='CompanyPeriodAccountValue',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.DecimalField(decimal_places=2, default=0, max_digits=18)),
                ('account', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='account_values', to='introml.Accounts')),
                ('company', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='company_values', to='introml.Companies')),
                ('period', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='period_values', to='introml.Periods')),
            ],
            options={
                'verbose_name': 'value',
                'verbose_name_plural': 'values',
            },
            bases=(models.Model, machinelearning.apps.introml.models.ModifyModel),
        ),
        migrations.CreateModel(
            name='CompanyPeriodAccountGeneral',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('value', models.CharField(blank=True, default='', max_length=250, null=True)),
                ('account', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='account_general_values', to='introml.Accounts')),
                ('company', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='account_general_values', to='introml.Companies')),
                ('period', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='period_general_values', to='introml.Periods')),
            ],
            options={
                'verbose_name': 'value',
                'verbose_name_plural': 'values',
            },
            bases=(models.Model, machinelearning.apps.introml.models.ModifyModel),
        ),
    ]
