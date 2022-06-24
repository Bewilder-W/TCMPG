from django.http import HttpResponse
from django.shortcuts import render
import json
from server.settings import BASE_DIR

# from sqlalchemy import JSON
# from .recommend_prescriptions import rp
import os


def index(request):
    return render(request, 'index.html')

# 接受前端发来的数据
def check(request):
    data = request.GET
    # data = JSON.stringify(data)
    print(data)
    base_path = str(BASE_DIR)
    path = os.path.join(BASE_DIR, "server/recommend_prescriptions.py")
    symptoms = data['symptoms']
    print(symptoms)

    cmd = "python " + path + " -symptoms " + symptoms + " -filepath " + base_path
    # cmd = "python " + path + " -symptoms " + symptoms
    prescription = os.popen(cmd).read()

    print(prescription)
    # rp(symptoms=symptoms, device='cpu')
    return HttpResponse(json.dumps({
        'prescription': prescription,
    }))