#-*- ecoding:utf-8 -*-
'''该脚本XXXX'''
import unicodecsv
## Longer version of code (replaced with shorter, equivalent version below)
# enrollments = []
# f = open('enrollments.csv', 'rb')
# reader = unicodecsv.DictReader(f)
# for row in reader:
#     enrollments.append(row)
# f.close()
# print(enrollments[0])
# with open('enrollments.csv', 'rb') as f:
#     reader = unicodecsv.DictReader(f)
#     enrollments = list(reader)
# print(enrollments[0])
### Write code similar to the above to load the engagement
### and submission data. The data is stored in files with
### the given filenames. Then print the first row of each
### table to make sure that your code works. You can use the
### "Test Run" button to see the output of your code.

enrollments_filename = 'enrollments.csv'
engagement_filename = 'daily_engagement.csv'
submissions_filename = 'project_submissions.csv'
def read_csv(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)
enrollments = read_csv(enrollments_filename)
daily_engagement = read_csv(engagement_filename)
project_submissions = read_csv(submissions_filename)

# print(enrollments[0])
# print(engagement[0])
# print(submissions[0])

from datetime import datetime as dt


# Takes a date as a string, and returns a Python datetime object.
# If there is no date given, returns None
def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%Y-%m-%d')


# Takes a string which is either an empty string or represents an integer,
# and returns an int or None.
def parse_maybe_int(i):
    if i == '':
        return None
    else:
        return int(i)


# Clean up the data types in the enrollments table
for enrollment in enrollments:
    enrollment['cancel_date'] = parse_date(enrollment['cancel_date'])
    enrollment['days_to_cancel'] = parse_maybe_int(enrollment['days_to_cancel'])
    enrollment['is_canceled'] = enrollment['is_canceled'] == 'True'
    enrollment['is_udacity'] = enrollment['is_udacity'] == 'True'
    enrollment['join_date'] = parse_date(enrollment['join_date'])

# print(enrollments[0])

for engagement_record in daily_engagement:
    engagement_record['lessons_completed'] = int(float(engagement_record['lessons_completed']))
    engagement_record['num_courses_visited'] = int(float(engagement_record['num_courses_visited']))
    engagement_record['projects_completed'] = int(float(engagement_record['projects_completed']))
    engagement_record['total_minutes_visited'] = float(engagement_record['total_minutes_visited'])
    engagement_record['utc_date'] = parse_date(engagement_record['utc_date'])

# print(daily_engagement[0])
# Clean up the data types in the submissions table
for submission in project_submissions:
    submission['completion_date'] = parse_date(submission['completion_date'])
    submission['creation_date'] = parse_date(submission['creation_date'])

# print(project_submissions[0])

enrollment_num_rows = len(enrollments)
engagement_num_rows = len(daily_engagement)
submission_num_rows = len(project_submissions)
print("---------total-----------")
print(enrollment_num_rows,engagement_num_rows,submission_num_rows)
# 统一命名
for item in daily_engagement:
    item['account_key'] = item['acct']
    del(item['acct'])

def get_unique_data(data):
    res = set()
    for item in data:
        res.add(item['account_key'])
    return res
# for enrollment in enrollments:
#     enrollment_acc.append(enrollment['account_key'])
# for engagement_record in daily_engagement:
#     engagement_acc.append(engagement_record['acct'])
# for submission in project_submissions:
#     submission_acc.append(submission['account_key'])
print("---------unique-----------")
unique_enrollment = get_unique_data(enrollments)
unique_engagement = get_unique_data(daily_engagement)
unique_submission = get_unique_data(project_submissions)
print(len(unique_enrollment),len(unique_engagement),len(unique_submission))

print('-------------------------------------------')

# print(daily_engagement[0])
# missing data
# print('missing data:')
# for enrollment in enrollments:
#     if enrollment['account_key'] not in unique_engagement:
#         print(enrollment['account_key'])
print('-------------------------------------------')
print('check for more problem records')
num_problem_students = 0
for enrollment in enrollments:
    student = enrollment['account_key']
    if (student not in unique_engagement and
            enrollment['join_date'] != enrollment['cancel_date']):
        print(enrollment)
        num_problem_students += 1
print(num_problem_students)
print('-------------------------------------------')
test_account = set()
for en in enrollments:
    if en['is_udacity']:
        test_account.add(en['account_key'])
print(len(test_account))

def remove_test_data(data):
    res = []
    for item in data:
        if item['account_key'] not in test_account:
            res.append(item)
    return res
no_test_enrollments = remove_test_data(enrollments)
no_test_engagement = remove_test_data(daily_engagement)
no_test_submissions = remove_test_data(project_submissions)
print('-------------------no udacity-----------------------')
print(len(no_test_enrollments),len(no_test_engagement),len(no_test_submissions))
print('-------------------------------------------------------')
paid_students = {}
for enrollment in no_test_enrollments:
    if (not enrollment['is_canceled'] or
            enrollment['days_to_cancel'] > 7):
        account_key = enrollment['account_key']
        enrollment_date = enrollment['join_date']
        if (account_key not in paid_students or
                enrollment_date > paid_students[account_key]):
        # if (enrollment_date > paid_students[account_key] or
        #             account_key not in paid_students):
        #这里写反的话,由于account_key不一定存在,paid_students[account_key]是错的
            paid_students[account_key] = enrollment_date
print(len(paid_students))