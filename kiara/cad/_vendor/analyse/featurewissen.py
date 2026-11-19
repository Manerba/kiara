"""CUST_FEAT_SETTINGS_V1: two context-bound native feature settings.

Read-only partial view; no CUST_FEAT/CF_REF/label/transform release. Native Creo
20.3 references isolated WARNING5->1 and FILING1->3/5/9/15. A separate inquiry
uniquely bound 15 source features to16 occurrences in one production source.
Private context is explicit, pinned by SHA and never embedded in the repository.
After native feature binding, featurebeschriftungen may additionally prove
five independently tested annotation parameters. Its whole-set graph guards
are separate; unknown label forms never revoke valid WARNING/FILING evidence.
Numeric rest fields stay raw.
"""
from dataclasses import dataclass, field
import collections
import hashlib
import json
import math
import re
import struct

from pkgleser import ist_zahl, zahl, segment_lesen, Segment, Objekt
from analyse.protokollfelder import tabelle_lesen, Tabellenfehler
from analyse.versuchsartefakte import ID_PATTERN
from objstrom import tokens, Unbekannt, SUB80, KURZ_ID_LO, Token

REGEL_ID = 'cust_feat_einstellungen_v1'
QUELLE = {'datei':'analyse/featurewissen.py','abschnitt':'CUST_FEAT_SETTINGS_V1'}
CUST_SHAPE = ['ID','B','v','v','NIL','D','c','c','I','B','NIL'] + ['N']*9 + ['B']*3 + ['N']*4 + ['B']*22 + ['LINK','LINK','v','v','B']

class Unbelegt(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _SegmentBeleg:
    """Internal provenance: constructed only after raw definitions and parsed graph agree."""
    seg: object
    fingerprint: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self,'fingerprint',_graph_integrity(self.seg))


def _typed_bytes(value,active=None,depth=0):
    """Immutable typed encoding; float bits retain NaN payload/sign and -0.

    Only tokenizer-compatible scalar/container values are accepted. The
    encoding is length-framed and distinguishes bool/int/float and tuple/list.
    No repr of caller-controlled objects and no mutable value is retained.
    """
    require(depth<=64,'Tokenwert_Verschachtelung_unbelegt')
    if value is None:return b'n'
    if type(value) is bool:return b'b1' if value else b'b0'
    if type(value) is int:
        magnitude=abs(value);length=(magnitude.bit_length()+7)//8
        return b'i'+(b'-' if value<0 else b'+')+struct.pack('>Q',length)+magnitude.to_bytes(length,'big')
    if type(value) is float:return b'f'+struct.pack('>d',value)
    if type(value) is str:
        data=value.encode('utf-8','surrogatepass');return b's'+struct.pack('>Q',len(data))+data
    if type(value) is bytes:return b'y'+struct.pack('>Q',len(value))+value
    if type(value) in (tuple,list):
        active=set() if active is None else active
        require(id(value) not in active,'Tokenwert_Zyklus_unbelegt')
        active.add(id(value))
        try:
            return (b't' if type(value) is tuple else b'l')+struct.pack('>Q',len(value))+b''.join(
                _typed_bytes(item,active,depth+1) for item in value)
        finally:active.remove(id(value))
    raise Unbelegt('Tokenwerttyp_unbelegt')


def _token_fingerprint(token):
    require(type(token) is Token,'Tokenobjekt_unbelegt')
    require(type(token.offset) is int and token.offset>=0,'Tokenoffset_unbelegt')
    require(type(token.art) is str,'Tokenarttyp_unbelegt')
    return hashlib.sha256(b'token-v1\0'+_typed_bytes((token.offset,token.art,token.wert))).digest()


def _fields_fingerprint(obj):
    require(type(obj.felder) is list,'Feldliste_unbelegt')
    digest=hashlib.sha256(b'fields-v1\0')
    for token in obj.felder:digest.update(_token_fingerprint(token))
    return len(obj.felder),digest.digest()


def _primary_id(obj):
    fs=fields(obj)
    return fs[0][1].wert if fs and fs[0][1].art=='ID' else None


def _graph_integrity(seg):
    require(type(seg) is Segment and type(seg.objekte) is dict,'Segmentobjekt_unbelegt')
    require(type(seg.start) is int and type(seg.ende) is int,'Segmentgrenze_unbelegt')
    require(type(seg.eintrag) is str and type(seg.klasse) is str and type(seg.id) is str,'Segmentkopf_Typ_unbelegt')
    require(seg.ende>seg.start>=0,'Segment_nicht_abgeschlossen')
    seen=set();rows=[]
    for nr,obj in seg.objekte.items():
        require(type(obj) is Objekt,'Graphobjekt_unbelegt')
        require(type(nr) is int and nr>=0 and type(obj.nr) is int and obj.nr==nr,'Objektnummer_mehrdeutig')
        require(type(obj.klasse) is str and type(obj.offset) is int and obj.offset>=0,'Objektkopf_Typ_unbelegt')
        fingerprint=_fields_fingerprint(obj)
        primary=_primary_id(obj)
        if primary is not None:
            require(type(primary) is str and primary not in seen,'Primaere_Objekt_ID_mehrdeutig')
            seen.add(primary)
        rows.append((nr,obj.klasse,obj.offset,primary,fingerprint))
    return seg.eintrag,seg.start,seg.ende,seg.klasse,seg.id,tuple(rows)


def _token_end(data,t):
    """Validate physical payload bounds, including sliced/truncated strings."""
    i,k=t.offset,t.art
    fixed={'R':6,'q':1,'r':1,'u':1,'I':1,'o':1,'U':3,'z':3,'t':7,'v':5,'B':5,'D':5,'O':2,'NIL':2,'F':9,'K':5,'ID':9}
    if k in fixed:return i+fixed[k]
    if k in ('c','d'):
        require(i+5<=len(data),'Tokenpayload_unvollstaendig')
        return i+5+2*int.from_bytes(data[i+1:i+5],'big')
    if k=='S':
        require(i+9<=len(data),'Tokenpayload_unvollstaendig')
        return i+9+2*int.from_bytes(data[i+5:i+9],'big')
    if k in ('a','s'):
        header=2 if k=='a' else 4
        require(i+header<=len(data),'Tokenpayload_unvollstaendig')
        return i+header+data[i+header-1]
    if k.startswith('80'):
        sub=int(k[2:],16);length=SUB80[sub]
        if sub in (0x11,0x12,0x14):
            require(i+3<=len(data),'Tokenpayload_unvollstaendig')
            length+=KURZ_ID_LO[data[i+2]]-2
        return i+2+length
    raise Unbelegt('Tokenform_unbelegt')


def _raw_index(data):
    """No dictionary collapse: inspect every t definition before segment_lesen."""
    result=[];active=None;obj=None;header=False;previous_end=0
    for token in tokens(data):
        require(token.offset==previous_end,'Tokenabgrenzung_unvollstaendig')
        previous_end=_token_end(data,token)
        require(previous_end<=len(data),'Tokenpayload_unvollstaendig')
        kind=token.art
        if kind=='q':
            require(active is None,'Segment_nicht_abgeschlossen')
            active=dict(start=token.offset,ende=None,klasse='',id='',objects=[],numbers=set(),ids=set(),features=0)
            obj=None;header=True
        elif active is None:
            require(kind not in ('t','u','r'),'Objekt_ausserhalb_Segment')
        elif kind=='r':
            require(not header and obj is None,'Objekt_oder_Segmentkopf_unvollstaendig')
            active['ende']=token.offset
            result.append(active);active=None
        elif header:
            require(kind not in ('t','u'),'Segmentkopf_unvollstaendig')
            if kind=='z':
                require(not active['klasse'],'Segmentkopf_mehrdeutig')
                active['klasse']=token.wert
            elif kind=='ID':
                require(not active['id'],'Segmentkopf_mehrdeutig')
                active['id']=token.wert
            elif kind=='8022':
                require(bool(active['klasse']) and bool(active['id']),'Segmentkopf_unvollstaendig')
                header=False
        elif kind=='t':
            require(obj is None,'Objekt_nicht_abgeschlossen')
            number,klasse=token.wert
            require(type(number) is int and number>=0 and number not in active['numbers'],'Objektnummer_mehrdeutig')
            require(isinstance(klasse,str) and not klasse.startswith('?'),'Objektklasse_nicht_aufgeloest')
            active['numbers'].add(number)
            active['features']+=klasse=='CUST_FEAT'
            obj=dict(nr=number,klasse=klasse,offset=token.offset,primary=None,first=True,
                     field_hasher=hashlib.sha256(b'fields-v1\0'),field_count=0)
            active['objects'].append(obj)
        elif kind=='u':
            require(obj is not None,'Objektende_ohne_Definition')
            obj['field_fingerprint']=(obj['field_count'],obj.pop('field_hasher').digest())
            obj=None
        elif obj is not None:
            obj['field_hasher'].update(_token_fingerprint(token));obj['field_count']+=1
            if kind not in ('s','8002') and obj['first']:
                obj['first']=False
                if kind=='ID':
                    require(isinstance(token.wert,str) and token.wert not in active['ids'],'Primaere_Objekt_ID_mehrdeutig')
                    active['ids'].add(token.wert);obj['primary']=token.wert
    require(previous_end==len(data) and active is None,'Segment_nicht_abgeschlossen')
    return result


def segmente_sicher(data,entry,warnings):
    """Return validated segments only. Any lexical stream damage blocks its entry."""
    try:
        index=_raw_index(data)
    except (Unbelegt,Unbekannt,struct.error,IndexError,UnicodeError,ValueError,KeyError) as exc:
        code=str(exc) if isinstance(exc,Unbelegt) else 'Objektstrom_nicht_vollstaendig_lesbar'
        warnings.append(dict(code=code,eintrag=entry,detail=str(exc)))
        return [],[code]
    out=[];errors=[]
    for raw_seg in index:
        local_warnings=[]
        try:
            seg=segment_lesen(data,raw_seg['start'],entry,local_warnings)
            require(not local_warnings and seg is not None,'Segmentleser_meldet_Warnung')
            require((seg.start,seg.ende,seg.klasse,seg.id)==(raw_seg['start'],raw_seg['ende'],raw_seg['klasse'],raw_seg['id']),
                    'Segmentkopf_oder_Grenze_widerspruechlich')
            require(seg.eintrag==entry,'Segmenteintrag_widerspruechlich')
            proof=_SegmentBeleg(seg)
            rows=proof.fingerprint[-1]
            expected=tuple((r['nr'],r['klasse'],r['offset'],r['primary']) for r in raw_seg['objects'])
            require(tuple(row[:4] for row in rows)==expected,'Objektdefinitionen_widerspruechlich')
            require(tuple(row[4] for row in rows)==tuple(r['field_fingerprint'] for r in raw_seg['objects']),
                    'Objektfelder_widerspruechlich')
            out.append((seg,proof))
        except (Unbelegt,Unbekannt,struct.error,IndexError,UnicodeError,ValueError,KeyError) as exc:
            code=str(exc) if isinstance(exc,Unbelegt) else 'Segment_nicht_vollstaendig_lesbar'
            errors.append(code)
            warnings.append(dict(code=code,eintrag=entry,start=raw_seg['start'],detail=str(exc),leserwarnungen=local_warnings))
    return out,errors


def require(value, reason):
    if not value:
        raise Unbelegt(reason)


def fields(obj):
    return [(i,t) for i,t in enumerate(obj.felder) if t.art not in ('s','8002')]


def finite(token):
    if not ((token.art == 'O' and type(token.wert) is int)
            or (token.art == 'F' and type(token.wert) in (int,float))):
        return False
    if not ist_zahl(token):
        return False
    try:
        return math.isfinite(zahl(token))
    except OverflowError:
        return False


def shape(obj, expected, table=False):
    pairs = fields(obj)
    if table:
        try:
            tab = tabelle_lesen(obj)
        except Tabellenfehler:
            raise Unbelegt('Tabellenform_unbelegt') from None
        require(tab is not None, 'Tabellenpraefix_fehlt')
        pairs = pairs[tab['ende']:]
    require(len(pairs) == len(expected), 'Objektform_unbelegt')
    for (_,token), kind in zip(pairs, expected):
        require((finite(token) if kind == 'N' else token.art in ('v','NIL') if kind == 'LINK' else token.art == kind), 'Feldform_unbelegt')
        if token.art in ('v','B','D'):
            require(type(token.wert) is int, 'Ganzzahlform_unbelegt')
        if token.art in ('ID','c','z'):
            require(isinstance(token.wert,str), 'Stringform_unbelegt')
        if token.art == 'NIL':
            require(type(token.wert) is int and token.wert == 1, 'NIL_Form_unbelegt')
    return pairs


def target(seg, pair, klasse):
    token = pair[1]
    require(token.art == 'v' and type(token.wert) is int, 'Verweisform_unbelegt')
    obj = seg.objekte.get(token.wert)
    require(obj is not None and obj.klasse == klasse, 'Verweisziel_unbelegt')
    return obj


def incoming(seg, obj, expected):
    actual = collections.Counter(owner.nr for owner in seg.objekte.values() for nr in owner.verweise() if nr == obj.nr)
    require(actual == collections.Counter(owner.nr for owner in expected), 'Weitere_oder_fehlende_Anheftung')


def source(obj, pair):
    i,token = pair
    filtered = next(j for j,(raw_index,_) in enumerate(fields(obj)) if raw_index == i)
    return dict(klasse=obj.klasse,objekt=obj.nr,feld=i,gefiltertes_feld=filtered,offset=token.offset)


def raw(value):
    if isinstance(value,float) and not math.isfinite(value):
        return {'nicht_endlich':repr(value)}
    if isinstance(value,(tuple,list)):
        return [raw(v) for v in value]
    return value


# The private schema is deliberately fixed to this one partial role rule.
_SHA = re.compile(r'[a-f0-9]{64}')
_ID = re.compile(ID_PATTERN)
_ARTIFACTS = {'source_pkg':'pkg_container','source_stream':'object_stream',
              'native_inquiry':'file','native_execution':'file','source_matching':'file'}
_WARNING = {'INVALIDATE':1,'INVALIDATE_CONFLICT':5}
_FILING = {'SD_FILE':1,'SD_MI':3,'SD_VRML':5,'SD_ANNOTATOR':9,'ALL_FOUR':15}
_FORM_O = tuple('O' if k=='N' else 'v' if k=='LINK' else k for k in CUST_SHAPE)
_FORM_F23 = _FORM_O[:23]+('F',)+_FORM_O[24:]


@dataclass(frozen=True,slots=True)
class _Stelle:
    segment_start:int
    segment_id:str
    object_number:int
    primary_id:str
    body_number:int
    body_id:str
    form:str
    warning_raw_index:int
    warning_offset:int
    filing_raw_index:int
    filing_offset:int
    warning_setting:str
    filing_setting:str
    native_occurrences:tuple


@dataclass(frozen=True,slots=True)
class Belegkontext:
    dokument_sha256:str
    package_sha256:str
    stream_sha256:str
    entry:str
    stellen:tuple
    artifact_refs:tuple


@dataclass(frozen=True,slots=True)
class Feldbeleg:
    object_number:int
    object_class:str
    object_offset:int
    fields_fingerprint:tuple
    rollen:tuple


def _keys(value, names):
    require(type(value) is dict and set(value)==set(names),'Kontext_Schluessel_unbelegt')


def _integer(value):
    require(type(value) is int and value>=0,'Kontext_Ganzzahl_unbelegt')


def _string(value):
    require(type(value) is str and bool(value),'Kontext_String_unbelegt')


def kontext_laden(data, *, erwarteter_sha256):
    """Load a previously reviewed, explicitly SHA-pinned PRIVATE context.

    This validates identity/schema, not the scientific truth of arbitrary JSON.
    Native evidence review remains external; no path is read or command run.
    """
    require(type(data) is bytes and type(erwarteter_sha256) is str and _SHA.fullmatch(erwarteter_sha256),'Kontext_Hashform_unbelegt')
    require(hashlib.sha256(data).hexdigest()==erwarteter_sha256,'Kontext_Hashabweichung')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'Kontext_Doppelter_Schluessel');out[k]=v
        return out
    def constant(_):raise Unbelegt('Kontext_Nichtendliche_Zahl')
    try:doc=json.loads(data,object_pairs_hook=pairs,parse_constant=constant)
    except (UnicodeError,json.JSONDecodeError) as error:raise Unbelegt('Kontext_JSON_unbelegt') from error
    _keys(doc,('schema_version','rule_id','artifacts','entry','sites'))
    require(type(doc['schema_version']) is int and doc['schema_version']==1 and doc['rule_id']==REGEL_ID,'Kontext_Version_oder_Regel_unbelegt')
    _keys(doc['artifacts'],_ARTIFACTS);refs=[];ids={}
    for name,scope in _ARTIFACTS.items():
        item=doc['artifacts'][name];_keys(item,('artifact_id','sha256','byte_scope'))
        require(type(item['artifact_id']) is str and _ID.fullmatch(item['artifact_id']) and not item['artifact_id'].startswith('slot.'),'Kontext_Artefakt_ID_unbelegt')
        require(type(item['sha256']) is str and _SHA.fullmatch(item['sha256']) and item['byte_scope']==scope,'Kontext_Artefaktbindung_unbelegt')
        value=(item['sha256'],scope)
        require(item['artifact_id'] not in ids or ids[item['artifact_id']]==value,'Kontext_Artefaktkollision')
        ids[item['artifact_id']]=value;refs.append((name,item['artifact_id'],item['sha256'],scope))
    _string(doc['entry']);require(type(doc['sites']) is list and bool(doc['sites']),'Kontext_Stellen_unbelegt')
    sites=[];seen=set();native_seen=set()
    names=('segment_start','segment_id','object_number','primary_id','body_number','body_id','form',
           'warning_raw_index','warning_offset','filing_raw_index','filing_offset','warning_setting','filing_setting','native_occurrences')
    for row in doc['sites']:
        _keys(row,names)
        for name in ('segment_start','object_number','body_number','warning_raw_index','warning_offset','filing_raw_index','filing_offset'):_integer(row[name])
        for name in ('segment_id','primary_id','body_id'):_string(row[name])
        require(all(type(row[name]) is str for name in ('form','warning_setting','filing_setting')),'Kontext_Einstellungsform_unbelegt')
        require(row['form'] in ('cust54_o','cust54_f23') and row['warning_setting'] in _WARNING and row['filing_setting'] in _FILING,'Kontext_Form_oder_Einstellung_unbelegt')
        require(type(row['native_occurrences']) is list and bool(row['native_occurrences']),'Kontext_Nativebeleg_fehlt')
        occurrences=[]
        for occurrence in row['native_occurrences']:
            _keys(occurrence,('case_id','feature_line','annotation_line'))
            require(type(occurrence['case_id']) is str and _ID.fullmatch(occurrence['case_id']),'Kontext_Nativefall_unbelegt')
            _integer(occurrence['feature_line']);_integer(occurrence['annotation_line'])
            require(occurrence['feature_line']>0 and occurrence['annotation_line']>0,'Kontext_Nativezeile_unbelegt')
            record=(occurrence['case_id'],occurrence['feature_line'],occurrence['annotation_line'])
            require(record not in occurrences,'Kontext_Doppelter_Nativebeleg');occurrences.append(record)
            require(record[:2] not in native_seen,'Kontext_Nativezuordnung_mehrdeutig');native_seen.add(record[:2])
        key=(row['segment_start'],row['segment_id'],row['object_number'],row['primary_id'])
        require(key not in seen,'Kontext_Doppelte_Stelle');seen.add(key)
        sites.append(_Stelle(*(row[name] for name in names[:-1]),tuple(occurrences)))
    return Belegkontext(erwarteter_sha256,doc['artifacts']['source_pkg']['sha256'],
                       doc['artifacts']['source_stream']['sha256'],doc['entry'],tuple(sites),tuple(refs))


def _owner(seg,cust,cp):
    body=target(seg,cp[2],'body')
    require(len(seg.von_klasse('body'))==1,'Besitzerkoerper_mehrdeutig')
    try:table=tabelle_lesen(body)
    except Tabellenfehler:raise Unbelegt('Besitzertabelle_unbelegt') from None
    require(table is not None,'Besitzertabelle_fehlt')
    found=[]
    for row in table['eintraege']:
        link=seg.objekte.get(row['ziel']) if row['art']=='v' else None
        if link is None or link.klasse!='ATTRIBUTE_LINK':continue
        fs=fields(link)
        if len(fs)<4 or fs[3][1].art!='z' or fs[3][1].wert!='CUST_FEAT':continue
        require(len(fs)>=6 and [t.art for _,t in fs[:5]]==['ID','B','v','z','B'],'Featureliste_Form_unbelegt')
        require(type(fs[1][1].wert) is int and fs[1][1].wert==1 and type(fs[2][1].wert) is int and fs[2][1].wert==body.nr,'Featureliste_Besitzer_unbelegt')
        require(type(fs[4][1].wert) is int and fs[4][1].wert==len(fs)-5,'Featureliste_Zahl_unbelegt')
        values=[]
        for pair in fs[5:]:
            other=target(seg,pair,'CUST_FEAT');require(other.nr not in values,'Featureliste_Doppeltes_Ziel');values.append(other.nr)
            other_fields=shape(other,CUST_SHAPE)
            require(target(seg,other_fields[2],'body') is body,'Featureliste_Rueckverweis_unbelegt')
        incoming(seg,link,[body])
        if cust.nr in values:found.append(link)
    require(len(found)==1,'Featureliste_Anheftung_mehrdeutig')
    # Reject any second or uninterpreted ATTRIBUTE_LINK path to this feature.
    actual=[obj for obj in seg.von_klasse('ATTRIBUTE_LINK') if cust.nr in obj.verweise()]
    require(len(actual)==1 and actual[0] is found[0],'Featureliste_Weitere_Anheftung')
    return body,found[0]


def rollen(seg,cust,*,integritaet,stelle):
    """PRIVATE detail view, called after the source-bound stream precheck."""
    try:
        require(type(integritaet) is _SegmentBeleg and integritaet.seg is seg,'Segmentbeleg_fehlt')
        require(integritaet.fingerprint==_graph_integrity(seg),'Segment_nach_Pruefung_veraendert')
        require(type(cust) is Objekt and seg.objekte.get(cust.nr) is cust,'Featureobjekt_nicht_identisch')
        require(type(stelle) is _Stelle,'Native_Stelle_fehlt')
        require((seg.start,seg.id,cust.nr,cust.id)==(stelle.segment_start,stelle.segment_id,stelle.object_number,stelle.primary_id),'Native_Stelle_abweichend')
        require(cust.klasse=='CUST_FEAT','Featureklasse_unbelegt')
        cp=shape(cust,CUST_SHAPE)
        require(tuple(t.art for _,t in cp)==(_FORM_O if stelle.form=='cust54_o' else _FORM_F23),'Vollstaendige_Featureform_unbelegt')
        body,link=_owner(seg,cust,cp)
        require((body.nr,body.id)==(stelle.body_number,stelle.body_id),'Nativer_Besitzer_abweichend')
        for filtered,index,offset in ((32,stelle.warning_raw_index,stelle.warning_offset),(33,stelle.filing_raw_index,stelle.filing_offset)):
            require((cp[filtered][0],cp[filtered][1].offset)==(index,offset),'Native_Feldquelle_abweichend')
        definitions=((32,_WARNING[stelle.warning_setting],'warning_setting','Gespeicherte native WARNING-Einstellung'),
                     (33,_FILING[stelle.filing_setting],'filing_setting','Gespeicherte native FILING-Einstellung'))
        roles=[]
        for index,expected,name,meaning in definitions:
            raw_index,t=cp[index]
            require(t.art=='B' and type(t.wert) is int and t.wert==expected,'Native_Einstellung_abweichend')
            roles.append({'rolle':name,'bedeutung':meaning,'wert_roh':t.wert,'quelle':source(cust,cp[index]),
                          'register_index':sum(x.art!='s' for x in cust.felder[:raw_index])})
        used={r['quelle']['feld'] for r in roles}
        return {'pruefung':'validiert','rollen':roles,'vollstaendige_feature_semantik':False,
                'besitzer':{'body':body.nr,'body_id':body.id,'attribute_link':link.nr},
                'restfelder_roh':[{'feld':i,'offset':t.offset,'art':t.art,'wert':raw(t.wert)} for i,t in enumerate(cust.felder) if i not in used]}
    except Unbelegt as error:
        return {'pruefung':str(error),'rollen':[],'vollstaendige_feature_semantik':False}


def strombelege(data,entry,*,quellpaket_sha256,belegkontext):
    """Small registry handoff: immutable per-object proofs and public status counts.

    Caller supplies the SHA of its actual outer package. The context loader pins
    the previously reviewed native report; this function binds its exact stream.
    No customer source or context path is implicit. Missing context stays open.
    """
    if type(belegkontext) is not Belegkontext:return {},{'belegkontext_fehlt':1}
    if (type(data) is not bytes or entry!=belegkontext.entry or quellpaket_sha256!=belegkontext.package_sha256
            or hashlib.sha256(data).hexdigest()!=belegkontext.stream_sha256):return {},{'belegquelle_abweichend':1}
    warnings=[];segments,errors=segmente_sicher(data,entry,warnings)
    if errors or warnings:return {},{'segmentintegritaet_unbelegt':1}
    results={};counts=collections.Counter();seen=set()
    sites={(s.segment_start,s.segment_id,s.object_number,s.primary_id):s for s in belegkontext.stellen}
    for seg,proof in segments:
        native_features=set()
        for cust in seg.von_klasse('CUST_FEAT'):
            key=(seg.start,seg.id,cust.nr,cust.id);site=sites.get(key)
            if site is None:counts['native_stelle_fehlt']+=1;continue
            seen.add(key);result=rollen(seg,cust,integritaet=proof,stelle=site);counts[result['pruefung']]+=1
            if result['pruefung']=='validiert':
                native_features.add(cust.nr)
                results[(cust.offset,cust.nr)]=Feldbeleg(cust.nr,cust.klasse,cust.offset,_fields_fingerprint(cust),
                    tuple((r['register_index'],r['bedeutung']) for r in result['rollen']))
        if native_features:
            from analyse.featurebeschriftungen import parameterbelege
            results.update(parameterbelege(seg,native_features))
    if set(sites)!=seen:return {},{'native_stelle_nicht_im_strom':len(set(sites)-seen)}
    return results,dict(counts)


def registerbelege(beleg,obj):
    """No raw values/IDs in the result; indexes omit s and retain8002."""
    out={'pruefung':'featurebeleg_fehlt','felder':[],'teilbits':[]}
    if type(beleg) is not Feldbeleg:return out
    try:
        require(type(obj) is Objekt and (obj.nr,obj.klasse,obj.offset)==(beleg.object_number,beleg.object_class,beleg.object_offset),'Featureobjekt_abweichend')
        require(_fields_fingerprint(obj)==beleg.fields_fingerprint,'Featurefelder_nach_Pruefung_veraendert')
    except Unbelegt as error:out['pruefung']=str(error);return out
    return {'pruefung':'validiert','felder':[{'index':i,'status':'belegt','bedeutung':meaning} for i,meaning in beleg.rollen],'teilbits':[]}
