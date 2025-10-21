from django.shortcuts import render

def index(request):
    """Main overview page"""
    context = {
        'title': 'Trustworthy Face Recognition System - Project Overview',
        'project_name': 'Face Recognition with Occlusion Handling'
    }
    return render(request, 'index.html', context)