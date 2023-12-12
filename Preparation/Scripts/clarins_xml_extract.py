import xmltodict
import pandas as pd
import os
import re

def special_characters(s, pat=re.compile('[@_!#$%^&*()<>?/\|}{~:]')):
    return pat.search(s)

# NETPATIENT
def fetch_xml(dir, source):
    df = pd.DataFrame(columns=['source', 'title', 'text_specialchars', 'text_without_specialchars', 'word_count'])
    
    for filename in os.listdir(dir):
        if filename.endswith('.xml'):
            print(filename)
            with open(os.path.join(dir, filename)) as file:
                my_xml = file.read()
                my_dict = xmltodict.parse(my_xml)
                df2 = pd.DataFrame.from_dict(my_dict, orient='index')
                
                title = df2['teiHeader'].iloc[0]['fileDesc']['titleStmt']['title']['#text']
                str_specialchars = ""
                str_chars = ""
                word_count = 0
            
                for i in df2["text"]['TEI']['spanGrp'][0]['span']:
                    try:
                        if special_characters(i['#text']) == None :
                            word_count += 1
                            str_chars += i['#text'] + " "
                            
                        str_specialchars += i['#text'] + " "
                        
                    except KeyError:
                       # print("KeyError")
                        None
                        
                df = df._append({'source': source,
                                'title': title,
                                'text_specialchars': str_specialchars, 
                                'text_without_specialchars': str_chars,
                                'word_count': word_count}, ignore_index=True)
          
    return df   

# main function
def main():
    df_netpatient = fetch_xml('/netpatient', "NetPatient")
    df_sundheddk = fetch_xml("/sundhedDK", "SundhedDK")
    df_regionh = fetch_xml("/regionH", "RegionH")
    df_naturvidenskab = fetch_xml("/AktuelNaturvidenskab", "AktuelNaturvidenskab")
    df_libris = fetch_xml("/libris_sundhed", "Libris")
    df_sst = fetch_xml("/Sundhed/SST", "SST")
    df_soefart = fetch_xml("/soefartsstyrelsen", "Søfart")
    
    BigDF = pd.DataFrame(columns=['source', 'title', 'text_specialchars', 'text', 'word_count'])
    BigDF = pd.concat([df_netpatient, df_sundheddk, df_regionh, df_naturvidenskab, df_libris, df_sst, df_soefart])
    BigDF.to_csv('BigDF.csv', index=False)

main()



