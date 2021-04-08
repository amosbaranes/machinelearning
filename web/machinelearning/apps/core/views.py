from django.shortcuts import render


def index(request):
    arg = {'title': 'Core'}
    return render(request, 'core/index.html', arg)

