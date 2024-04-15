from pathlib import Path

import PySimpleGUI as sg

import pandas as pd

import numpy as np

import os

import cv2

import threading

import time

import datetime

from datetime import datetime

counting = 0

def newdba():

 folder_path = f'D:/DATABASE/DBA'

 if not os.path.exists(folder_path):

 os.makedirs(folder_path)

 print(f"Folder created: {folder_path}")

 else:

 print(f"Folder already exists: {folder_path}")

 excel_file_path = f'D:/DATABASE/DBA/Data.xlsx'

 if not os.path.exists(excel_file_path):

 data = {'User ID': [1]}

 df = pd.DataFrame(data)

 df.to_excel(excel_file_path, index=False)

 print(f"Excel file created: {excel_file_path}")

 else:

 df = pd.read_excel(excel_file_path)

 if 'User ID' not in df.columns:

 # If 'User ID' column is not present, create it

 df['User ID'] = 1

 user_id = df['User ID'].max() + 1

 df.loc[df.index.max() + 1] = {'User ID': user_id}

 df.to_excel(excel_file_path, index=False)

 print(f"User ID added to Excel file: {excel_file_path}")

 user_id = df['User ID'].max()

 sg.popup(f"Your user ID is {user_id}")

 sg.popup("See In The Camera")

 time.sleep(2)

 camera = cv2.VideoCapture(0)

 _, frame = camera.read()

 camera.release()

 if not os.path.exists(folder_path):
os.makedirs(folder_path)

 photo_file_name = f'D:/DATABASE/DBA/User-{user_id}.jpg'

 cv2.imwrite(photo_file_name, frame)

 sg.popup('Data saved!')

def schedule(Class, Year, Section, excel_file_path):

 def clear_input(window, values):

 for key in values:

 window[key]('')

 return None

 Class = Class

 Year = Year

 Section = Section

 sg.theme('DarkTeal9')

 Layout = [

 [sg.Text('Enter The Number Of Schedules')],

 [sg.Text('Schedule', size=(15, 1)), sg.InputText(key='Schedule')],

 [sg.Submit(), sg.Button('Clear')]

 ]

 window = sg.Window('Enter Number Of Schedules', Layout)

 event, values = window.read()

 if event == 'Clear':

 clear_input(window, values)

 if event == 'Submit':

 Schedule = int(values['Schedule'])

 for _ in range(1, Schedule + 1):

 Layout = [

 [sg.Text(f'Enter Schedule {_} Details')],

 [sg.Text('Lecture_Name', size=(15, 1)), 

sg.InputText(key='Lecture_Name'),

 sg.Text('Time_IN_HH:MM', size=(15, 1)), 

sg.InputText(key='Time_In')],

 [sg.Submit(), sg.Button('Clear')]

 ]

 window = sg.Window(f'Enter Schedule {_} Details', Layout)

 event, values = window.read()

 if event == 'Submit':

 Time = values['Time_In']

 Lecture = values['Lecture_Name']

 EXCEL_FILE = excel_file_path

 Path(EXCEL_FILE).parent.mkdir(parents=True, exist_ok=True)

 if Path(EXCEL_FILE).exists():

 df_data_entry = pd.read_excel(EXCEL_FILE)

 else:

 df_data_entry = pd.DataFrame()

 new_record = pd.DataFrame({Lecture: [Time]})
df_data_entry = pd.concat([df_data_entry, new_record], 

ignore_index=True)

 df_data_entry.to_excel(EXCEL_FILE, index=False)

 # change the format of data

 df = pd.read_excel(excel_file_path)

 df = df.apply(lambda col: col.ffill().bfill())

 df = df.drop_duplicates(keep='first')

 df.to_excel(excel_file_path, index=False)

 df = pd.read_excel(excel_file_path)

 df = df.apply(lambda col: col.ffill().bfill())

 df = df.drop_duplicates(keep='first')

 df.to_excel(excel_file_path, index=False)

 window.close()

def change_schedule():

 def clear_input():

 for key in values:

 window[key]('')

 return None

 layout = [

 [sg.Text('Please fill out the following fields:')],

 [sg.Text('Class', size=(15, 1)),

 sg.Combo(['B.C.A.', 'B.SC.', 'B.COM.', 'B.B.A.', 'B.A.', 'FASHION 

TECHNOLOGY.'], key='Class')],

 [sg.Text('Year', size=(15, 1)), sg.Combo(['I', 'II', 'III'], 

key='Year')],

 [sg.Text('Section', size=(15, 1)), sg.Combo(['A', 'B', 'C'], 

key='Section')],

 [sg.Submit(), sg.Button('Clear')]

 ]

 window = sg.Window('Simple data entry form', layout)

 current_dir = Path(__file__).parent if '__file__' in locals() else 

Path.cwd()

 event, values = window.read()

 if event == 'Clear':

 clear_input(window, values)

 if event == 'Submit':

 Class = values['Class']

 Year = values['Year']

 Section = values['Section']

 excel_file_path = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/Schedule.xlsx'

 if not os.path.exists(excel_file_path):

 schedule(Class, Year, Section)

 else:

 sg.theme('DarkTeal9')

 Layout = [

 [sg.Text('Enter The Number Of Schedules')],
[sg.Text('Schedule', size=(15, 1)), 

sg.InputText(key='Schedule')],

 [sg.Submit(), sg.Button('Clear')]

 ]

 window = sg.Window('Enter Number Of Schedules', Layout)

 event, values = window.read()

 if event == 'Submit':

 Schedule = int(values['Schedule'])

 for _ in range(1, Schedule + 1):

 Layout = [

 [sg.Text(f'Enter Schedule {_} Details')],

 [sg.Text('Lecture_Name', size=(15, 1)), 

sg.InputText(key='Lecture_Name'),

 sg.Text('Time_IN_HH:MM', size=(15, 1)), 

sg.InputText(key='Time_In')],

 [sg.Submit(), sg.Button('Clear')]

 ]

 window = sg.Window(f'Enter Schedule {_} Details', Layout)

 event, values = window.read()

 if event == 'Submit':

 Time = values['Time_In']

 Lecture = values['Lecture_Name']

 EXCEL_FILE = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/schedule.xlsx'

 Path(EXCEL_FILE).parent.mkdir(parents=True, 

exist_ok=True)

 # Clear existing data before adding new data

 df_data_entry = pd.DataFrame()

 new_record = pd.DataFrame({Lecture: [Time]})

 df_data_entry = pd.concat([df_data_entry, 

new_record], ignore_index=True)

 df_data_entry.to_excel(EXCEL_FILE, index=False)

 # change the format of data

 df = pd.read_excel(excel_file_path)

 df = df.apply(lambda col: col.ffill().bfill())

 df = df.drop_duplicates(keep='first')

 df.to_excel(excel_file_path, index=False)

 df = pd.read_excel(excel_file_path)

 df = df.apply(lambda col: col.ffill().bfill())

 df = df.drop_duplicates(keep='first')

 df.to_excel(excel_file_path, index=False)

 clear_input(window, values)

 window.close()

def change_password():

 def clear_input(window, values):

 for key in values:
window[key]('')

 return None

 excel_file_path = 'D:/DATABASE/DBA/Password.xlsx'

 if not os.path.exists(excel_file_path):

 sg.theme('DarkTeal9')

 layout = [

 [sg.Text('Enter The Password')],

 [sg.Text('Password', size=(15, 1)), 

sg.InputText(key='Password')],

 [sg.Text('Confirm_Password', size=(15, 1)), 

sg.InputText(key='Confirm_Password')],

 [sg.Submit(), sg.Button('Clear')]

 ]

 window = sg.Window('Enter Password', layout)

 event, values = window.read()

 if event == 'Submit':

 Password = values['Password']

 Confirm_Password = values['Confirm_Password']

 if Password == Confirm_Password:

 data = {'Password': [Password]}

 df = pd.DataFrame(data)

 df.to_excel(excel_file_path, index=False)

 else:

 sg.popup("Password and Confirm_Password are not the same")

 change_password()

 if event == 'Clear':

 clear_input(window, values)

 else:

 sg.theme('DarkTeal9')

 layout = [

 [sg.Text('Enter The Password')],

 [sg.Text('Current_Password', size=(15, 1)), 

sg.InputText(key='Current_Password')],

 [sg.Text('New_Password', size=(15, 1)), 

sg.InputText(key='New_Password')],

 [sg.Text('Confirm_Password', size=(15, 1)), 

sg.InputText(key='Confirm_Password')],

 [sg.Submit(), sg.Button('Clear')]

 ]

 window = sg.Window('Enter Password', layout)

 event, values = window.read()

 if event == 'Submit':

 df = pd.read_excel(excel_file_path)

 Current_Password = values['Current_Password']

 New_Password = values['New_Password']

 Confirm_Password = values['Confirm_Password']

 if df['Password'].iloc[0] == Current_Password:

 if New_Password == Confirm_Password:

 data = {'Password': [New_Password]}

 df = pd.DataFrame(data)

 df.to_excel(excel_file_path, index=False)

 else:
sg.popup("Password and Confirm_Password are not the 

same")

 change_password()

 else:

 sg.popup("Your Entered Current_Password Is Incorrect")

 change_password()

 if event == 'Clear':

 clear_input(window, values)

 window.close()

def registration():

 def clear_input(window, values):

 for key in values:

 window[key]('')

 return None

 sg.theme('DarkTeal9')

 Layout = [

 [sg.Text('Enter The Password')],

 [sg.Text('Password', size=(15, 1)), sg.InputText(key='Password')],

 [sg.Submit(), sg.Button('Clear')]

 ]

 window = sg.Window('Enter Password', Layout)

 event, values = window.read()

 if event == 'Submit':

 excel_file_path = f'D:/DATABASE/DBA/Password.xlsx'

 if not os.path.exists(excel_file_path):

 change_password()

 else:

 Password = values['Password']

 df = pd.read_excel(excel_file_path)

 if Password == df['Password'].iloc[0]:

 sg.popup("Authentication Successful")

 else:

 sg.popup("Wrong Password")

 registration()

 if event == 'Clear':

 clear_input(window, values)

 layout = [

 [sg.Text('Please fill out the following fields:')],

 [sg.Text('Name', size=(15, 1)), sg.InputText(key='Name')],

 [sg.Text('Phone_Number', size=(15, 1)), 

sg.InputText(key='Phone_Number')],

 [sg.Text('Class', size=(15, 1)),

 sg.Combo(['B.C.A.', 'B.SC', 'B.COM', 'B.B.A.', 'B.A.', 'FASHION 

TECHNOLOGY'], key='Class')],

 [sg.Text('Year', size=(15, 1)), sg.Combo(['I', 'II', 'III'], 

key='Year')],

 [sg.Text('Section', size=(15, 1)), sg.Combo(['A', 'B', 'C'], key='Section')],

 [sg.Submit(), sg.Button('Clear'), sg.Exit(), sg.Button('Login'), 

sg.Button('Change_Password'),

 sg.Button('New_DBA'), sg.Button('change_schedule')]

 ]

 window = sg.Window('Simple data entry form', layout)

 current_dir = Path(__file__).parent if '__file__' in locals() else 

Path.cwd()

 while True:

 event, values = window.read()

 if event == 'change_schedule':

 change_schedule()

 if event == 'Change_Password':

 change_password()

 if event == 'New_DBA':

 newdba()

 if event == sg.WIN_CLOSED or event == 'Exit':

 break

 if event == 'Login':

 login()

 if event == 'Clear':

 clear_input(window, values)

 if event == 'Submit':

 Name = values['Name']

 Class = values['Class']

 Year = values['Year']

 Section = values['Section']

 Phone_Number = values['Phone_Number']

 folder_path = f'D:/DATABASE/{Class}/{Class}-{Year}/{Section}'

 if not os.path.exists(folder_path):

 os.makedirs(folder_path)

 print(f"Folder created: {folder_path}")

 else:

 print(f"Folder already exists: {folder_path}")

 excel_file_path = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/Data.xlsx'

 if not os.path.exists(excel_file_path):

 data = {'User ID': [1]}

 df = pd.DataFrame(data)

 df.to_excel(excel_file_path, index=False)

 print(f"Excel file created: {excel_file_path}")

 else:

 df = pd.read_excel(excel_file_path)

 user_id = df['User ID'].max() + 1
df.loc[df.index.max() + 1] = {'User ID': user_id}

 df.to_excel(excel_file_path, index=False)

 print(f"User ID added to Excel file: {excel_file_path}")

 user_id = df['User ID'].max()

 sg.popup(f"Your user ID is {user_id}")

 EXCEL_FILE = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/Data_Entry.xlsx'

 Path(EXCEL_FILE).parent.mkdir(parents=True, exist_ok=True)

 if Path(EXCEL_FILE).exists():

 df_data_entry = pd.read_excel(EXCEL_FILE)

 else:

 df_data_entry = pd.DataFrame()

 new_record = pd.DataFrame(

 {'User ID': [user_id], 'Name': [Name], 'Class': [Class], 

'Year': [Year], 'Section': [Section],

 'Phone_Number': [Phone_Number]})

 df_data_entry = pd.concat([df_data_entry, new_record], 

ignore_index=True)

 df_data_entry.to_excel(EXCEL_FILE, index=False)

 time.sleep(3)

 sg.popup("See In The Camera")

 time.sleep(2)

 camera = cv2.VideoCapture(0)

 _, frame = camera.read()

 camera.release()

 folder_path = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/Photos'

 if not os.path.exists(folder_path):

 os.makedirs(folder_path)

 Name = Name.replace(" ", "_")

 photo_file_name = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/Photos/User-{user_id}.jpg'

 cv2.imwrite(photo_file_name, frame)

 sg.popup('Data saved!')

 clear_input(window, values)

 excel_file_path = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/Schedule.xlsx'

 if not os.path.exists(excel_file_path):

 schedule(Class, Year, Section, excel_file_path)

 window.close()

def attendance(Class, Year, Section, user_id):

 user_id = user_id

 Class = Class

 Year = Year

 Section = Section

 excel_file_path_schedule = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/schedule.xlsx'

 # Create directories recursively if they don't exist
directory = os.path.dirname(excel_file_path_schedule)

 if not os.path.exists(directory):

 os.makedirs(directory)

 if not os.path.exists(excel_file_path_schedule):

 new_record = pd.DataFrame({'User ID': [user_id], 'Class': [Class], 

'Year': [Year], 'Section': [Section]})

 df_data_entry = new_record # Define df_data_entry here

 df_data_entry.to_excel(excel_file_path_schedule, index=False)

 df_schedule = pd.read_excel(excel_file_path_schedule)

 df_schedule = df_schedule.apply(lambda col: col.ffill().bfill())

 df_schedule = df_schedule.drop_duplicates(keep='first')

 current_time = datetime.now().time() # Moved outside the loop to get the 

current time only once

 df_schedule = df_schedule.astype(str)

 matching_columns = []

 print("Current Time:", current_time)

 print("Schedule Data:")

 print(df_schedule)

 for col in df_schedule.columns:

 for time_range in df_schedule[col]:

 # Check if the cell contains a valid time range

 if '-' in time_range:

 start_time, end_time = time_range.split('-')

 start_time = start_time.strip()

 end_time = end_time.strip()

 # Convert start_time and end_time to datetime objects

 start_time_dt = datetime.strptime(start_time, "%H:%M").time() 

# Convert to time object

 end_time_dt = datetime.strptime(end_time, "%H:%M").time() # 

Convert to time object

 # Adjust end_time if it's less than start_time (spanning two 

days)

 if end_time_dt < start_time_dt:

 end_time_dt = (datetime.combine(datetime.today(), 

end_time_dt) + timedelta(days=1)).time()

 # Check if the current time falls within the time range with 

a 10-minute grace period

 if start_time_dt <= current_time <= 

(datetime.combine(datetime.today(), start_time_dt) + 

timedelta(minutes=10)).time():

 matching_columns.append(col)

 break # Break the loop if a match is found

 if matching_columns:

 excel_file_path_attendance = f'D:/DATABASE/{Class}/{Class}-

{Year}/{Section}/attendance.xlsx'

 # Read existing attendance data or create an empty DataFrame if the 

file doesn't exist

 if Path(excel_file_path_attendance).exists():
df_data_entry = pd.read_excel(excel_file_path_attendance)

 else:

 df_data_entry = pd.DataFrame()

 # Check if the 'User ID' column is not in the DataFrame or the 

DataFrame is empty

 if 'User ID' not in df_data_entry.columns or df_data_entry.empty:

 # Create 'User ID' column and initialize with NaN values

 df_data_entry['User ID'] = None

 # Check if the user_id already exists in the DataFrame

 if user_id not in df_data_entry['User ID'].values:

 # Initialize a new record for the user_id

 new_record = {'User ID': user_id}

 # Set default values for all columns to empty string

 for col in df_schedule.columns:

 new_record[col] = 0 # Default to 0 for all subjects

 for matching_column in matching_columns:

 new_record[matching_column] = 1 # Mark the matched subject 

as attended (1)

 # Append the new record to the DataFrame

 df_data_entry = pd.concat([df_data_entry, 

pd.DataFrame([new_record])], ignore_index=True)

 # Save the updated DataFrame to the Excel file

 df_data_entry.to_excel(excel_file_path_attendance, index=False)

 print(f"Attendance recorded for User ID: {user_id}")

 else:

 print(f"Attendance already recorded for User ID: {user_id}")

 else:

 print(f"No data found for the current time: {current_time}")

def DBA_result(result):

 if result:

 print("Authentication successful!")

 registration()

 else:

 print("Authentication failed.")

 login()

def Login_result(result, Class, Year, Section, user_id):

 if result:

 print("Authentication successful!")

 attendance(Class, Year, Section, user_id)

 else:

 print("Authentication failed.")

 login()

def login():
sg.theme('DarkTeal9')

 layout = [

 [sg.Text('Please fill out the following fields:')],

 [sg.Text('Class', size=(15, 1)),

 sg.Combo(['B.C.A', 'B.SC', 'B.COM', 'B.B.A', 'B.A', 'FASHION 

TECHNOLOGY'], key='Class')],

 [sg.Text('Year', size=(15, 1)), sg.Combo(['I', 'II', 'III'], 

key='Year')],

 [sg.Text('Section', size=(15, 1)), sg.Combo(['A', 'B', 'C'], 

key='Section')],

 [sg.Submit(), sg.Button('Clear'), sg.Exit()]

 ]

 window = sg.Window('Student Attendance', layout, finalize=True)

 event, values = window.read()

 if event == 'Clear':

 clear_input()

 if event == 'Submit':

 Class = values['Class']

 Year = values['Year']

 Section = values['Section']

 folder_path = f'D:/DATABASE/{Class}/{Class}.-{Year}/{Section}/Photos'

 files = [f for f in os.listdir(folder_path) if 

os.path.isfile(os.path.join(folder_path, f))]

 user_ids = len(files)

 print(f"There are {user_ids} files in the folder.")

 face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 

'haarcascade_frontalface_default.xml')

 for user_id in range(1, user_ids + 1):

 if user_id == user_ids + 1:

 login()

 print("Be patient, it takes some time")

 cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

 cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

 reference_img = cv2.imread(f'D:/DATABASE/{Class}/{Class}.-

{Year}/{Section}/Photos/User-{user_id}.jpg',

 cv2.IMREAD_GRAYSCALE)

 def check_face(frame):

 try:

 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 faces = face_cascade.detectMultiScale(gray, 

scaleFactor=1.3, minNeighbors=5)

 if len(faces) > 0:

 x, y, w, h = faces[0]

 current_face = gray[y:y + h, x:x + w]

 reference_face = cv2.resize(reference_img, (w, h))

 return True
else:

 return False

 except Exception as e:

 print(f"Error: {e}")

 return False

 for i in range(1, 11):

 ret, frame = cap.read()

 if ret:

 if threading.active_count() < 2:

 result = check_face(frame.copy())

 Login_result(result, Class, Year, Section, user_id)

 if check_face(frame):

 cap.release()

 cv2.destroyAllWindows()

 return True

 else:

 print("Authentication error!!!")

 cv2.imshow("video", frame)

 if cv2.waitKey(1) & 0xFF == ord('q'):

 break

 window.close()

def DBA():

 folder_path = f'D:/DATABASE/DBA'

 if not os.path.exists(folder_path):

 newdba()

 registration()

 else:

 files = [f for f in os.listdir(folder_path) if 

os.path.isfile(os.path.join(folder_path, f))]

 # Get the count of files

 user_id = len(files)

 print(f"There are {user_id} files in the folder.")

 face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 

'haarcascade_frontalface_default.xml')

 for user_id in range(1, user_id + 1):

 print("Be patient, it takes some time")

 cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

 cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

 reference_img = cv2.imread(f'D:/DATABASE/DBA/User-{user_id}.jpg',

 cv2.IMREAD_GRAYSCALE)

 def check_face(frame):

 try:

 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 faces = face_cascade.detectMultiScale(gray, 

scaleFactor=1.3, minNeighbors=5)
if len(faces) > 0:

 x, y, w, h = faces[0]

 current_face = gray[y:y + h, x:x + w]

 reference_face = cv2.resize(reference_img, (w, h))

 # Dummy implementation - always consider it a match

 return True

 else:

 return False

 except Exception as e:

 print(f"Error: {e}")

 return False

 for i in range(1, 11):

 ret, frame = cap.read()

 if ret:

 if threading.active_count() < 2:

 result = check_face(frame.copy())

 DBA_result(result)

 if check_face(frame):

 cap.release()

 cv2.destroyAllWindows()

 return True

 else:

 print("Authentication error!!!")

 cv2.imshow("video", frame)

 if cv2.waitKey(1) & 0xFF == ord('q'):

 break

 cap.release()

 cv2.destroyAllWindows()

 return False

def main():

 DBA()

if __name__ == "__main__":

 main()
