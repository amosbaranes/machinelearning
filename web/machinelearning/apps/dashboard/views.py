from django.http import JsonResponse
from django.shortcuts import render
from .models import Order
from django.core import serializers


def dashboard_with_pivot(request):
    print(1)
    return render(request, 'dashboard/dashboard_with_pivot.html', {})


def pivot_data(request):
    dataset = Order.objects.all()
    data = serializers.serialize('json', dataset)
    print(data)
    return JsonResponse(data, safe=False)
