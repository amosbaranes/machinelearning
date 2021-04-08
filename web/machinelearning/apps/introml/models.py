from django.db import models
from django.db import connection


class ModifyModel(object):

    @classmethod
    def truncate(cls):
        with connection.cursor() as cursor:
            cursor.execute('TRUNCATE TABLE {} RESTART IDENTITY CASCADE'.format(cls._meta.db_table))


class Accounts(models.Model, ModifyModel):

    class Meta:
        verbose_name = 'account'
        verbose_name_plural = 'accounts'
        ordering = ['category']

    category = models.CharField(max_length=50, default='', blank=True, null=True)
    account = models.CharField(max_length=250, default='', blank=True, null=True)
    account_name = models.CharField(max_length=250, default='', blank=True, null=True)

    def __str__(self):
        return str(self.category) + ': ' + str(self.account_name)


class Companies(models.Model, ModifyModel):

    class Meta:
        verbose_name = 'company'
        verbose_name_plural = 'companies'
        ordering = ['cik']

    cik = models.CharField(max_length=20, default='', blank=True, null=True)

    def __str__(self):
        return str(self.cik)


class Periods(models.Model, ModifyModel):

    class Meta:
        verbose_name = 'period'
        verbose_name_plural = 'periods'

    year_quarter = models.IntegerField(primary_key=True)
    year = models.SmallIntegerField(default=0)
    quarter = models.SmallIntegerField(default=0)

    def __str__(self):
        return str(self.year) + ' ' + str(self.quarter)


class CompanyPeriodAccountGeneral(models.Model, ModifyModel):
    class Meta:
        verbose_name = 'CompanyPeriodAccountGeneral'
        verbose_name_plural = 'CompanyPeriodAccountGenerals'

    company = models.ForeignKey(Companies, null=True, on_delete=models.CASCADE, related_name='account_general_values')
    period = models.ForeignKey(Periods, null=True, on_delete=models.CASCADE, related_name='period_general_values')
    account = models.ForeignKey(Accounts, null=True, on_delete=models.CASCADE, related_name='account_general_values')
    value = models.CharField(max_length=1020, default='', blank=True, null=True)


class CompanyPeriodAccountValue(models.Model, ModifyModel):

    class Meta:
        verbose_name = 'CompanyPeriodAccountValue'
        verbose_name_plural = 'CompanyPeriodAccountValues'

    company = models.ForeignKey(Companies, null=True, on_delete=models.CASCADE, related_name='company_values')
    period = models.ForeignKey(Periods, null=True, on_delete=models.CASCADE, related_name='period_values')
    account = models.ForeignKey(Accounts, null=True, on_delete=models.CASCADE, related_name='account_values')
    value = models.DecimalField(max_digits=21, decimal_places=2, default=0)

