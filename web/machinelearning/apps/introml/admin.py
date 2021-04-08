from django.contrib import admin
from .models import (Accounts, Companies, Periods, CompanyPeriodAccountGeneral, CompanyPeriodAccountValue)


@admin.register(Accounts)
class AccountsAdmin(admin.ModelAdmin):
    list_display = ('id', 'category', 'account', 'account_name')
    list_filter = ('category', )


@admin.register(Companies)
class CompaniesAdmin(admin.ModelAdmin):
    list_display = ('id', 'cik')


@admin.register(Periods)
class PeriodsAdmin(admin.ModelAdmin):
    list_display = ('year', 'quarter')


@admin.register(CompanyPeriodAccountGeneral)
class CompanyPeriodAccountGeneralAdmin(admin.ModelAdmin):
    list_display = ('id', 'company', 'period', 'account', 'value')
    list_filter = ('company', 'period', 'account')


@admin.register(CompanyPeriodAccountValue)
class CompanyPeriodAccountValueAdmin(admin.ModelAdmin):
    list_display = ('id', 'company', 'period', 'account', 'value')
    list_filter = ('company', 'period', 'account')


