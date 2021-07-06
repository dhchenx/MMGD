import csv
import pandas as pd


class MMObj:

    def __init__(self, objectId, objectPath, objectType):
        self.objectId = objectId
        self.objectPath = objectPath
        self.objectType = objectType

class MMLabel:
    def __init__(self, labelId, objectId,object=None):
        self.labelId = labelId
        self.objectId = objectId
        self.object=object

class MMMode:
    def __init__(self, modeId, labelId,label=None):
        self.modeId = modeId
        self.labelId = labelId
        self.label=label


class MMNode:
    def __init__(self, nodeId, labelId, modeId,label=None,mode=None):
        self.nodeId = nodeId
        self.labelId = labelId
        self.modeId = modeId
        self.label=label
        self.mode=mode


class MMMNode:
    def __init__(self, mnodeId):
        self.mnodeId = mnodeId


class MMMMNode:
    def __init__(self, mmnodeId):
        self.mmnodeId = mmnodeId

class MMGDFiles:

    def __init__(self,root_path):
        self.root_path=root_path
        self.label_ids_path=self.root_path+"/labels/ids.txt"
        self.node_ids_path = self.root_path + "/nodes/ids.txt"
        self.mnode_ids_path = self.root_path + "/mnodes/ids.txt"
        self.mmnode_ids_path = self.root_path + "/mmnodes/ids.txt"
        self.mode_ids_path = self.root_path + "/modes/ids.txt"
        self.model_ids_path = self.root_path + "/models/ids.txt"
        self.object_ids_path = self.root_path + "/objects/ids.txt"
        self.relation_ids_path = self.root_path + "/relations/ids.txt"
        self.property_ids_path = self.root_path + "/properties/ids.txt"

    def get_file_keys(self,file_path, id_field):
        keys = []
        with open(file_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                # print(row)
                keys.append(row[id_field])
        return keys

    def get_label_ids(self):
        return self.get_file_keys(self.label_ids_path,id_field='labelId')

    def get_node_ids(self):
        return self.get_file_keys(self.node_ids_path,id_field='nodeId')

    def get_mnode_ids(self):
        return self.get_file_keys(self.mnode_ids_path,id_field='mnodeId')

    def get_mmnode_ids(self):
        return self.get_file_keys(self.mmnode_ids_path,id_field='mmnodeId')

    def get_mode_ids(self):
        return self.get_file_keys(self.mode_ids_path,id_field='modeId')

    def get_property_ids(self):
        return self.get_file_keys(self.property_ids_path,id_field='propertyId')

    def get_relation_ids(self):
        return self.get_file_keys(self.relation_ids_path,id_field='relationId')

    def get_model_ids(self):
        return self.get_file_keys(self.model_ids_path,id_field='modelId')

    def get_object_ids(self):
        return self.get_file_keys(self.object_ids_path,id_field='objectId')

    def search_file_by_id(self,path,id_field,id_value,split=','):
        df_csv = pd.read_csv(path, sep=',')
        df_csv=df_csv.loc[(df_csv[id_field]==int(id_value))]
        dict_row=df_csv.to_dict('records')
        # print(dict_row)
        return dict_row

    def search_file_by_where(self,path,query_str):
        df_csv = pd.read_csv(path, sep=',')
        df_csv=df_csv.query(query_str)
        dict_row=df_csv.to_dict('records')
        # print(dict_row)
        return dict_row

    def get_single_label(self,label_id):

        pass

    def get_single_mode(self,mode_id):

        pass


    def get_single_node(self,node_id):
        if node_id not in self.get_node_ids():
            raise Exception('the node id does not exist in node id list!')
        node_info=self.search_file_by_where(self.node_ids_path,'nodeId=='+str(node_id))
        print('node',node_info)
        label_info = self.search_file_by_where(self.label_ids_path, 'labelId=='+str(node_info[0]['labelId']))
        print('label',label_info)
        mode_info = self.search_file_by_where(self.mode_ids_path, 'modeId==' + str(node_info[0]['modeId']))
        print('mode',mode_info)
        label_object_info=self.search_file_by_where(self.object_ids_path, 'objectId==' + str(label_info[0]['objectId']))
        print('label_obj',label_object_info)
        mode_label_info = self.search_file_by_where(self.label_ids_path,
                                                      'labelId==' + str(mode_info[0]['labelId']))

        print('mode_label',mode_label_info)
        mode_label_obj_info = self.search_file_by_where(self.object_ids_path,
                                                    'objectId==' + str(mode_label_info[0]['objectId']))

        print('mode_label_obj', mode_label_obj_info[0])

        label_model={}
        label_model["labelId"]=node_info[0]['labelId']

        label_model['objectId']=self.search_file_by_where(self.object_ids_path, 'objectId==' + str(label_info[0]['objectId']))[0]

        mode_label_model={}
        mode_label_model["labelId"]=mode_info[0]["labelId"]
        mode_label_model["objectId"] =mode_label_obj_info[0]["objectId"]
        mode_label_model["object"] = mode_label_obj_info[0]

        mode_model={}
        mode_model["modeId"]=mode_info[0]["modeId"]
        mode_model["label"]=mode_label_model
        mode_model["labelId"] = mode_label_model["labelId"]

        node_model={}
        node_model["nodeId"]=node_id
        node_model["labelId"] = label_model["labelId"]
        node_model["modeId"] = mode_model["modeId"]
        node_model["label"]=label_model
        node_model["mode"] = mode_model

        # construct mode model
        m_mode_label_obj=MMObj(mode_label_obj_info[0]["objectId"],mode_label_obj_info[0]["objectType"],mode_label_obj_info[0]["objectPath"])
        m_mode_label=MMLabel(mode_label_model["labelId"],mode_label_model["objectId"],m_mode_label_obj)
        m_mode=MMMode(mode_model["modeId"],mode_model["labelId"],m_mode_label)

        # construct label model
        m_label_obj=MMObj(label_object_info[0]["objectId"],label_object_info[0]["objectPath"],label_object_info[0]["objectType"])
        m_label=MMLabel(label_model["labelId"],m_label_obj.objectId,m_label_obj)

        # construct node model
        m_node=MMNode(node_model["nodeId"],node_model["labelId"],node_model["modeId"],m_label,m_mode)

        return m_node

mmgd_files=MMGDFiles(root_path='db')
print(mmgd_files.get_label_ids())
# print(mmgd_files.search_file(mmgd_files.node_ids_path,'nodeId','22',split=','))
print()
node=mmgd_files.get_single_node(node_id='1')

print(node)
