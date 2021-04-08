from django.conf.urls import url
from .views import (index, show_content, ch01, ch02, ch03, ch04, proj01, proj02, get_symbol_data, gold)

app_name = "introml"

urlpatterns = [
    url(r'^$', index, name='index'),
    url(r'^show_content/$', show_content, name='show_content'),
    url(r'^ch01/$', ch01, name='ch01'),
    url(r'^ch02/$', ch02, name='ch02'),
    url(r'^ch03/$', ch03, name='ch03'),
    url(r'^ch04/$', ch04, name='ch04'),
    #
    url(r'^proj01/$', proj01, name='proj01'),
    url(r'^proj02/$', proj02, name='proj02'),
    url(r'^get_symbol_data/$', get_symbol_data, name='get_symbol_data'),
    url(r'^gold/$', gold, name='gold'),
]


