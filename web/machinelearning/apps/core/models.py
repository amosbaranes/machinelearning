from django.conf import settings
from django.utils import timezone
from django.urls import reverse
from django.db import models


class SecurityGroup(models.Model):
    class Meta:
        verbose_name = 'security group'
        verbose_name_plural = 'security group'
        ordering = ['group']

    id = models.AutoField(primary_key=True)
    group = models.CharField(max_length=50, blank=True)


class Security(models.Model):
    class Meta:
        verbose_name = 'security'
        verbose_name_plural = 'securities'
        ordering = ['symbol']

    id = models.AutoField(primary_key=True)
    symbol = models.CharField(max_length=10, blank=True)
    security_name = models.CharField(max_length=128, blank=True)
    security_group = models.ForeignKey(SecurityGroup, on_delete=models.CASCADE, null=True, related_name='securities')


class Price(models.Model):
    class Meta:
        verbose_name = 'price'
        verbose_name_plural = 'prices'
        ordering = ('price_date', )

    security = models.ForeignKey(Security, on_delete=models.CASCADE, null=True, related_name='prices')
    price_date = models.DateField(null=True)
    open = models.DecimalField(max_digits=12, decimal_places=6, default=0)
    high = models.DecimalField(max_digits=12, decimal_places=6, default=0)
    low = models.DecimalField(max_digits=12, decimal_places=6, default=0)
    close = models.DecimalField(max_digits=12, decimal_places=6, default=0)
    adj_close = models.DecimalField(max_digits=12, decimal_places=6, default=0)
    volume = models.IntegerField(default=0)
    yy_mm_dd = models.IntegerField(default=0)

    def save(self, *args, **kwargs):
        if not self.yy_mm_dd:
            self.yy_mm_dd = self.price_date.year*10000 + self.price_date.month*100 + self.price_date.day
        super(Price, self).save(*args, **kwargs)



