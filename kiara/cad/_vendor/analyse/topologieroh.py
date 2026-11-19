"""Complete raw topology forms, with a separate directed-graph audit.

No BRep reader, geometry evaluation, repairs or register-role promotion. The
five own-form grammars and reference target classes are inventoried in exactly
two production originals. Names such as next/prev/partner are structural
candidates; static consistency does not establish their native geometric role.
Local snapshots retain every token, including declarations and numeric sentinels.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping
import copy
from dataclasses import dataclass

from objstrom import Token
from pkgleser import Objekt
from analyse.protokollfelder import Tabellenfehler, tabelle_lesen

KLASSEN=frozenset({'coedge','edge','face','loop','vertex'})
FLAECHEN=frozenset({'k2_bspline_sf','k2_cone','k2_cylinder','k2_helical_sf',
    'k2_parallel_swept_bspline','k2_plane','k2_sphere','k2_spun_bspline','k2_torus'})
KURVEN=frozenset({'k2_bspline_cv','k2_circle','k2_ellipse','k2_ssi_curve','k2_straight',
    'k2_transf_bspline_cv','k2_transf_cone_bspline_cv'})


class TopologieFormfehler(ValueError):
    def __init__(self,code,index=None):
        super().__init__(code)
        self.code=code
        self.index=index


@dataclass
class RohTopologie:
    nummer: int
    klasse: str
    offset: int
    rohfelder: tuple
    bereinigte_indizes: tuple
    praefix_ende: int
    nutzfeld_indizes: tuple
    gruppen: dict
    struktur: dict
    strukturrollen: tuple
    formgruppe: str
    semantik_freigegeben: bool = False


@dataclass
class SegmentTopologie:
    objekte: dict
    klassen: dict
    fehler: dict
    semantik_freigegeben: bool = False


def objekt_lesen(objekt,kontext):
    """Read one complete own form; snapshots are detached from the input.

    `praefix_ende` indexes fields after removing s/8002; `nutzfeld_indizes` and
    all group/role indices refer to the complete original field list instead.
    Numerical tokens are never converted, sorted, clamped or made finite.
    """
    if not isinstance(objekt,Objekt) or not isinstance(kontext,Mapping):
        raise TypeError('Objekt und lokales Objekt-Mapping erforderlich')
    if type(objekt.nr) is not int or objekt.nr<0 or kontext.get(objekt.nr) is not objekt:
        raise TopologieFormfehler('objekt_nicht_eindeutig_im_kontext')
    if objekt.klasse not in KLASSEN:raise TopologieFormfehler('andere_klasse')
    if any(not isinstance(t,Token) for t in objekt.felder):raise TopologieFormfehler('ungueltiges_token')
    try:prefix=tabelle_lesen(objekt)
    except Tabellenfehler as exc:raise TopologieFormfehler('d_praefix_ungueltig') from exc
    if prefix is None:raise TopologieFormfehler('d_praefix_fehlt')
    indices=tuple(i for i,t in enumerate(objekt.felder) if t.art not in ('s','8002'))
    own=indices[prefix['ende']:]
    fields=[objekt.felder[i] for i in own]
    groups={};roles=[];values={};pos=0

    def fail(code,i=None):raise TopologieFormfehler(code,own[i] if i is not None and i<len(own) else None)
    def take(name,count):
        nonlocal pos
        if type(count) is not int or count<0 or count>len(fields)-pos:fail('anzahl_ueberschreitet_rest',pos)
        start=pos;pos+=count;groups[name]=own[start:pos]
        return start,fields[start:pos]
    def integer(name,count=1):
        start,ts=take(name,count)
        for j,t in enumerate(ts):
            if t.art!='B' or type(t.wert) is not int:fail('b_ganzzahl_erwartet',start+j)
        return tuple(t.wert for t in ts)
    def count(name):
        n=integer(name)[0]
        if n<0:fail('negative_anzahl',pos-1)
        if n>len(fields)-pos:fail('anzahl_ueberschreitet_rest',pos-1)
        roles.append(dict(index=own[pos-1],rolle='anzahl_serialisierter_verweise',
                          status='strukturell_geprueft',liste=name.removesuffix('_anzahl')))
        return n
    def numeric(name,n):
        start,ts=take(name,n)
        for j,t in enumerate(ts):
            ok=(t.art=='F' and type(t.wert) in (int,float)) or (t.art=='O' and type(t.wert) is int and t.wert in (1,2,3,4))
            if not ok:fail('numerisches_token_erwartet',start+j)
        return tuple(copy.deepcopy(t) for t in ts)
    def references(name,classes,n=1,nil=False):
        if type(classes) is str:classes={classes}
        start,ts=take(name,n);out=[]
        for j,t in enumerate(ts):
            if nil and t.art=='NIL' and type(t.wert) is int and t.wert==1:
                out.append(None);continue
            if t.art!='v' or type(t.wert) is not int or t.wert<0:fail('lokaler_verweis_erwartet',start+j)
            target=kontext.get(t.wert)
            if target is None:fail('verweisziel_fehlt',start+j)
            if not isinstance(target,Objekt) or type(target.nr) is not int or target.nr!=t.wert:
                fail('kontextnummer_widerspruch',start+j)
            if target.klasse not in classes:fail('verweiszielklasse_unbeobachtet',start+j)
            out.append(t.wert)
            roles.append(dict(index=own[start+j],rolle='lokaler_objektverweis',
                zielklasse=target.klasse,status='strukturell_geprueft'))
        return tuple(out)
    def one(name,classes,nil=False):return references(name,classes,nil=nil)[0]

    cls=objekt.klasse
    if cls=='coedge':
        values['surface']=one('surface',FLAECHEN)
        values['uv_raw']=numeric('uv_raw',4)
        values['b5']=integer('b5')[0]
        values['edge']=one('edge','edge')
        values['pcurve']=one('pcurve','k2_straight2',nil=True)
        values['b8']=integer('b8')[0]
        values['parameter_box_raw']=numeric('parameter_box_raw',6)
        for name in ('next','prev'):values[name]=one(name,'coedge')
        values['partner']=one('partner','coedge',nil=True)
        values['loop']=one('loop','loop')
        kind='nil' if values['partner'] is None else 'self' if values['partner']==objekt.nr else 'other'
        form='coedge_partner_'+kind
    elif cls=='edge':
        values['curve']=one('curve',KURVEN,nil=True)
        if pos+2<=len(fields) and all(t.art=='O' and type(t.wert) is int and t.wert==5 for t in fields[pos:pos+2]):
            if values['curve'] is not None:fail('o5_intervall_mit_kurve',pos)
            _,ts=take('parameters_raw',2)
            values['parameters_raw']=tuple(copy.deepcopy(t) for t in ts)
            values['parameter_kind']='o5_o5'
        else:
            values['parameters_raw']=numeric('parameters_raw',2)
            values['parameter_kind']='numeric'
        values['b3_b6']=integer('b3_b6',4)
        values['vertices']=references('vertices','vertex',2)
        values['owner']=one('owner',{'coedge','wire'})
        values['owner_class']=kontext[values['owner']].klasse
        values['box_raw']=numeric('box_raw',6)
        if values['curve'] is None and values['owner_class']=='wire':fail('nil_kurve_wire_kontext_unbeobachtet')
        form='edge_'+('nil_'+values['parameter_kind'] if values['curve'] is None else 'curve')+'_'+values['owner_class']
    elif cls=='face':
        n=count('loops_anzahl')
        values['loops']=references('loops','loop',n)
        values['parent']=one('parent',{'shell','subshell'})
        values['surface']=one('surface',FLAECHEN)
        values['b_after_surface']=integer('b_after_surface')[0]
        values['box_raw']=numeric('box_raw',6)
        values['parameter_box_raw']=numeric('parameter_box_raw',6)
        values['tail_b']=integer('tail_b',5)
        form='face_parent_'+kontext[values['parent']].klasse
    elif cls=='loop':
        values['start']=one('start','coedge')
        values['face']=one('face','face')
        values['box_raw']=numeric('box_raw',6)
        values['parameter_box_raw']=numeric('parameter_box_raw',6)
        values['tail_b']=integer('tail_b',3)
        form='loop'
    else:
        n=count('edges_anzahl')
        values['edges']=references('edges','edge',n)
        values['point']=one('point','k2_point')
        form='vertex'
    if pos!=len(fields):fail('zusaetzliche_eigene_felder',pos)
    # Every own field belongs to exactly one named raw group, including opaque B.
    covered=[i for group in groups.values() for i in group]
    if sorted(covered)!=list(own) or len(set(covered))!=len(own):raise RuntimeError('interne_feldabdeckung')
    return RohTopologie(objekt.nr,cls,objekt.offset,tuple(copy.deepcopy(t) for t in objekt.felder),
        indices,prefix['ende'],own,groups,values,tuple(roles),form)


def segment_lesen(objekte):
    """Snapshot all five classes; reject duplicate identities before parsing."""
    if isinstance(objekte,(str,bytes,Mapping)):raise TypeError('Explizite Objektfolge erforderlich')
    objects={}
    for obj in objekte:
        if not isinstance(obj,Objekt) or type(obj.nr) is not int or obj.nr<0:
            raise TopologieFormfehler('ungueltiges_segmentobjekt')
        if obj.nr in objects:raise TopologieFormfehler('doppelte_objektnummer')
        objects[obj.nr]=obj
    parsed={};errors={}
    for nr,obj in objects.items():
        if obj.klasse not in KLASSEN:continue
        try:parsed[nr]=objekt_lesen(obj,objects)
        except TopologieFormfehler as exc:
            errors[nr]=dict(klasse=obj.klasse,code=exc.code,index=exc.index,
                rohfelder=tuple(copy.deepcopy(t) for t in obj.felder))
    return SegmentTopologie(parsed,{nr:o.klasse for nr,o in objects.items()},errors)


def graph_pruefen(snapshot):
    """Return local structural checks; an open/failed check never repairs data."""
    if not isinstance(snapshot,SegmentTopologie):raise TypeError('SegmentTopologie erforderlich')
    parsed=snapshot.objekte;out=[]
    opaque_coedges=[nr for nr,e in snapshot.fehler.items() if e['klasse']=='coedge']
    def check(rule,nr,targets,predicate,reason=None):
        missing=[t for t in targets if t not in parsed]
        status='offen' if missing else 'bestanden' if predicate() else 'widerspruch'
        out.append(dict(regel=rule,nummer=nr,status=status,ziele=tuple(targets),
                        grund='ziel_nicht_dekodiert' if missing else reason))
    byloop=defaultdict(set);byedge=defaultdict(set)
    for nr,o in parsed.items():
        if o.klasse=='coedge':
            byloop[o.struktur['loop']].add(nr)
            byedge[o.struktur['edge']].add(nr)
    for nr,o in parsed.items():
        p=o.struktur;cls=o.klasse
        if cls=='coedge':
            for direction,opposite in [('next','prev'),('prev','next')]:
                target=p[direction]
                check('coedge_'+direction+'_inverse',nr,[target],lambda t=target,op=opposite:parsed[t].struktur[op]==nr)
            check('coedge_next_same_loop',nr,[p['next']],lambda:parsed[p['next']].struktur['loop']==p['loop'])
            partner=p['partner']
            if partner is not None:
                check('coedge_partner_inverse',nr,[partner],lambda:parsed[partner].struktur['partner']==nr)
                check('coedge_partner_same_edge',nr,[partner],lambda:parsed[partner].struktur['edge']==p['edge'])
                if partner!=nr:
                    check('coedge_nonself_partner_opposite_binary_b5',nr,[partner],
                          lambda:{p['b5'],parsed[partner].struktur['b5']}=={0,1})
            if partner is None or partner==nr:
                check('coedge_special_partner_nil_curve_context',nr,[p['edge']],lambda:parsed[p['edge']].struktur['curve'] is None)
            loop=parsed.get(p['loop'])
            face=loop.struktur['face'] if loop else None
            check('coedge_surface_matches_loop_face',nr,[p['loop'],face],
                  lambda:parsed[face].struktur['surface']==p['surface'])
        elif cls=='edge':
            if p['owner_class']=='coedge':
                check('edge_coedge_backref',nr,[p['owner']],lambda:parsed[p['owner']].struktur['edge']==nr)
            else:
                check('wire_edge_has_no_coedges',nr,opaque_coedges,lambda:not byedge[nr])
            for vertex in p['vertices']:
                check('edge_vertex_incidence',nr,[vertex],lambda v=vertex:nr in parsed[v].struktur['edges'])
            if p['curve'] is None:
                check('nil_curve_same_vertex',nr,[],lambda:p['vertices'][0]==p['vertices'][1])
                check('nil_curve_one_coedge',nr,opaque_coedges,lambda:len(byedge[nr])==1)
        elif cls=='vertex':
            for edge in p['edges']:
                check('vertex_edge_endpoint_incidence',nr,[edge],lambda e=edge:nr in parsed[e].struktur['vertices'])
        elif cls=='face':
            for loop in p['loops']:
                check('face_loop_backref',nr,[loop],lambda l=loop:parsed[l].struktur['face']==nr)
        elif cls=='loop':
            check('loop_face_membership',nr,[p['face']],lambda:nr in parsed[p['face']].struktur['loops'])
            seen=set();current=p['start'];same_loop=True;missing=None
            while current not in seen:
                coedge=parsed.get(current)
                if coedge is None:
                    missing=current;break
                if coedge.klasse!='coedge':raise RuntimeError('interner_linktyp_widerspruch')
                seen.add(current)
                same_loop &= coedge.struktur['loop']==nr
                current=coedge.struktur['next']
            targets=[] if missing is None else [missing]
            check('loop_next_cycle_closes_at_start',nr,targets,lambda:current==p['start'] and same_loop)
            check('loop_cycle_covers_coedge_backrefs',nr,targets+opaque_coedges,lambda:seen==byloop[nr] and same_loop)
    return dict(pruefungen=out,semantik_freigegeben=False,
                grenze='Directed raw-link consistency; no native identity, normal, trim or manifold proof.')


def bericht(snapshot,graph=None):
    """Public class/form-group counters only, never raw IDs/values/locations."""
    if not isinstance(snapshot,SegmentTopologie):raise TypeError('SegmentTopologie erforderlich')
    classes=Counter(o.klasse for o in snapshot.objekte.values())
    forms=Counter(o.formgruppe for o in snapshot.objekte.values())
    errors=Counter((e['klasse'],e['code']) for e in snapshot.fehler.values())
    count_roles=Counter(r['rolle'] for o in snapshot.objekte.values() for r in o.strukturrollen)
    checks=defaultdict(Counter)
    if graph is not None:
        for c in graph['pruefungen']:checks[c['regel']][c['status']]+=1
    return dict(schema_version=1,semantik_freigegeben=False,klassen=dict(sorted(classes.items())),
        formgruppen=dict(sorted(forms.items())),
        formfehler=[dict(klasse=k,code=c,anzahl=n) for (k,c),n in sorted(errors.items())],
        strukturrollen=dict(sorted(count_roles.items())),
        graphpruefung={k:dict(sorted(v.items())) for k,v in sorted(checks.items())},
        grenzen=['Only structural counts/lists/references have been checked.',
                 'Opaque flags and numeric fields retain no native geometric role from this result.',
                 'Complete raw tokens exist only in the local snapshot.'])
