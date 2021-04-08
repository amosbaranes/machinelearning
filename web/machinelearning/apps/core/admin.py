from django.contrib import admin
from .models import Security, SecurityGroup, Price


@admin.register(Security)
class SecurityAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'security_name']


@admin.register(SecurityGroup)
class SecurityGroupAdmin(admin.ModelAdmin):
    list_display = ['id', 'group']


@admin.register(Price)
class PriceAdmin(admin.ModelAdmin):
    list_display = ['price_date', 'adj_close', 'volume']
