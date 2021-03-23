import nengo
import numpy as np
import nengo_spa as spa

D = 256
MAX_SIMILARITY = 0.1
WTA_NEURONS = 1000
CONVERSION_THRESHOLD = 0.4

pointers = ['DILEMMA_FB', 'DILEMMA_S', 'DILEMMA_L', 'TROLLEY_SCENE',
    'ACTION', 'GOOD_RESULT', 'BAD_RESULT','PROTOTYPICAL_VIOLENCE'
    'PUSH_BYSTANDER', 'BYSTANDER_FALLS', 'BYSTANDER_LANDS_ON_TRACK',
    'PULL_SWITCH', 'ALIGN_TRACK', 'TURN_TROLLEY',
    'UPSET_FAMILY', 'TROLLEY_ON_SECOND_TRACK',
    'RUNAWAY_TROLLEY = LIVES_LOST*FIVE + LIVES_SAVED*ONE',
    'STOP_TROLLEY_WITH_BYSTANDER = LIVES_LOST*ONE + HARM*PROTOTYPICAL_VIOLENCE',
    'TROLLEY_HITS_BYSTANDER = LIVES_LOST*ONE',
    'SAVE_FIVE_PEOPLE = LIVES_SAVED*FIVE',
    'APPROPRIATE = ACTION*GOOD_RESULT',
    'INAPPROPRIATE = ACTION*NEGATIVE',
#    'INAPPROPRIATE = PUSH_BYSTANDER*NEGATIVE',
]


#Sean's Code
vocab = spa.Vocabulary(D, strict=False, max_similarity=MAX_SIMILARITY)

#Sean's Code
vocab.populate(';'.join(i for i in pointers))
vocab.add('NULL', np.zeros(D))

vocab_emotion_valence = vocab.create_subset(['NEGATIVE', 'NEUTRAL_OR_POSITIVE'])

#Sean's Code    
conversion_dict = {vocab['NULL']: 0,
                   vocab['ONE']: 1, 
                   vocab['TWO']: 2, 
                   vocab['THREE']: 3, 
                   vocab['FOUR']: 4, 
                   vocab['FIVE']: 5,
                   }

entailmentMap = {
    
    'RUNAWAY_TROLLEY_UNDER_FOOTBRIDGE*PUSH_BYSTANDER':'BYSTANDER_FALLS',
    'RUNAWAY_TROLLEY_UNDER_FOOTBRIDGE*BYSTANDER_FALLS':'BYSTANDER_LANDS_ON_TRACKS',
    'RUNAWAY_TROLLEY_UNDER_FOOTBRIDGE*BYSTANDER_LANDS_ON_TRACKS':'TROLLEY_HITS_BYSTANDER',
    'RUNAWAY_TROLLEY_UNDER_FOOTBRIDGE*TROLLEY_HITS_BYSTANDER':'STOP_TROLLEY_WITH_BYSTANDER',
    'RUNAWAY_TROLLEY_UNDER_FOOTBRIDGE*STOP_TROLLEY_WITH_BYSTANDER':'SAVE_FIVE_PEOPLE',
    
    'RUNAWAY_TROLLEY_BYSTANDER_ON_SIDE_TRACK*PULL_SWITCH':'ALIGN_TRACK',
    'RUNAWAY_TROLLEY_BYSTANDER_ON_SIDE_TRACK*ALIGN_TRACK':'TURN_TROLLEY',
    'RUNAWAY_TROLLEY_BYSTANDER_ON_SIDE_TRACK*TURN_TROLLEY':'SAVE_FIVE_PEOPLE',
    
    'RUNAWAY_TROLLEY_BYSTANDER_ON_LOOP*PULL_SWITCH':'ALIGN_TRACK',
    'RUNAWAY_TROLLEY_BYSTANDER_ON_LOOP*ALIGN_TRACK':'TURN_TROLLEY',
    'RUNAWAY_TROLLEY_BYSTANDER_ON_LOOP*TURN_TROLLEY':'SAVE_FIVE_PEOPLE',
}

complexEntailmentMap = {

    'RUNAWAY_TROLLEY_BYSTANDER_ON_SIDE_TRACK*TROLLEY_ON_SECOND_TRACK':'TROLLEY_HITS_BYSTANDER',
    
    'RUNAWAY_TROLLEY_BYSTANDER_ON_LOOP*TROLLEY_ON_SECOND_TRACK':'STOP_TROLLEY_WITH_BYSTANDER',
    'RUNAWAY_TROLLEY_BYSTANDER_ON_LOOP*STOP_TROLLEY_WITH_BYSTANDER':'SAVE_FIVE_PEOPLE',
}



#Convert semantic pointer to number
def convert(t, x):
    """Converts semantic pointer labels to numerical values"""
    
    similarities = [np.dot(x, key.v) for key in conversion_dict]

    if max(similarities) > CONVERSION_THRESHOLD:
        return similarities.index(max(similarities))
    else:
        return 0

#Convert number to semantic pointer
def convert_flip(t, x):
    """Convers numerical representation of decision into semantic pointer vectors"""
    if x >= 0.9:
        return vocab['GOOD_RESULT'].v
    elif x <= -0.9:
        return vocab['BAD_RESULT'].v
    else:
        return vocab['NULL'].v


#Input for dilemas
def task_input(t):
    if t < 0.04:
        return 'TROLLEY_SCENE'
    elif 0.15 < t < 0.35:
        return 'DILEMMA_S'
    elif 1.0 < t < 1.04:
        return 'TROLLEY_SCENE'
    elif 1.15 < t < 1.35:
        return 'DILEMMA_FB'
    elif 2.0 < t < 2.04:
        return 'TROLLEY_SCENE'
    elif 2.15 < t < 2.35:
        return 'DILEMMA_L'
    else:
        return '0'

with spa.Network() as model:
    
    symbl = spa.sym
    
    #Sates and Transcodes
    input_transcode = spa.Transcode(task_input, output_vocab = vocab)
    input_state = spa.State(vocab)
    circumstance = spa.State(vocab, feedback = 0.65)
    lives_lost_state = spa.State(vocab)
    lives_saved_state = spa.State(vocab)
    primary_causal_chain = spa.State(vocab, feedback = 0.55)
    secondary_causal_chain = spa.State(vocab, feedback = 0.5)
    myopic_alarm = spa.State(vocab)
    emotional_state = spa.State(vocab)
    predicted_outcome = spa.State(vocab)
    judgment = spa.State(vocab)
    
    #Associative Memory
    Entailments = spa.ThresholdingAssocMem(threshold=0.3, input_vocab=vocab,
        mapping=entailmentMap, function=lambda x: x > 0)
        
    Complex_Entailments = spa.ThresholdingAssocMem(threshold=0.5, input_vocab=vocab,
        mapping=complexEntailmentMap, function=lambda x: x > 0)
        
    Clean_Up = spa.WTAAssocMem(threshold=0.225, input_vocab=vocab,
        mapping=vocab.keys(), function=lambda x: x > 0)  
        
    #Sean's Code
    lost_convert_node = nengo.Node(output=convert, size_in=D, size_out=1,
    label='Lives Lost')
    saved_convert_node = nengo.Node(output=convert, size_in=D, size_out=1,
    label='Lives Saved')
    
    nengo.Connection(lives_lost_state.output, lost_convert_node)
    nengo.Connection(lives_saved_state.output, saved_convert_node)
    
    merged_ens = nengo.Ensemble(200, 2, label='merged ensemble')
    compute_ens = nengo.Ensemble(100, 1, label='compute ensemble')
    
    nengo.Connection(saved_convert_node, merged_ens[0])
    nengo.Connection(lost_convert_node, merged_ens[1])
    nengo.Connection(merged_ens, compute_ens, function=lambda x: x[0] - x[1]) 
    
    convert_to_sp = nengo.Node(output=convert_flip, size_in=1, size_out=D, label='convert to sp')
    
    nengo.Connection(compute_ens, convert_to_sp)
    nengo.Connection(convert_to_sp, predicted_outcome.input)
        
    #Binding Operations
    circumstance * primary_causal_chain >> Entailments
    
    #Connecitons
    input_transcode >> input_state
    #Entailments >> primary_causal_chain
    nengo.Connection(Entailments.output, primary_causal_chain.input, synapse=0.0425)
    circumstance * secondary_causal_chain >> Complex_Entailments
    Complex_Entailments >> secondary_causal_chain
    circumstance *~ symbl.LIVES_SAVED >> lives_saved_state
    circumstance *~ symbl.LIVES_LOST >> lives_lost_state
    primary_causal_chain *~ symbl.HARM >> myopic_alarm
    primary_causal_chain *~ symbl.LIVES_SAVED >> lives_saved_state
    primary_causal_chain *~ symbl.LIVES_LOST >> lives_lost_state
    secondary_causal_chain *~ symbl.LIVES_SAVED >> lives_saved_state
    secondary_causal_chain *~ symbl.LIVES_LOST >> lives_lost_state
    emotional_state * symbl.ACTION >> judgment
    
 
    
    with spa.ActionSelection():
        spa.ifmax(spa.dot(input_state, symbl.TROLLEY_SCENE),
        symbl.RUNAWAY_TROLLEY >> circumstance)
        
        spa.ifmax(spa.dot(input_state, symbl.DILEMMA_S),
        symbl.PULL_SWITCH >> primary_causal_chain,
        symbl.RUNAWAY_TROLLEY_BYSTANDER_ON_SIDE_TRACK >> circumstance)
        
        spa.ifmax(spa.dot(input_state, symbl.DILEMMA_FB),
        symbl.PUSH_BYSTANDER >> primary_causal_chain,
        symbl.RUNAWAY_TROLLEY_UNDER_FOOTBRIDGE >> circumstance)
        
        spa.ifmax(spa.dot(input_state, symbl.DILEMMA_L),
        symbl.PULL_SWITCH >> primary_causal_chain,
        symbl.RUNAWAY_TROLLEY_BYSTANDER_ON_LOOP >> circumstance)
        
        spa.ifmax(spa.dot(primary_causal_chain, symbl.TURN_TROLLEY),
        symbl.TROLLEY_ON_SECOND_TRACK >> secondary_causal_chain)
        
        spa.ifmax(spa.dot(primary_causal_chain, symbl.STOP_TROLLEY_WITH_BYSTANDER),
        symbl.UPSET_FAMILY >> secondary_causal_chain)
        
        spa.ifmax(spa.dot(primary_causal_chain, symbl.SAVE_FIVE_PEOPLE),
        predicted_outcome * symbl.ACTION >> Clean_Up)
    
    with spa.ActionSelection():
        spa.ifmax(spa.dot(myopic_alarm, symbl.PROTOTYPICAL_VIOLENCE),
        symbl.NEGATIVE >> emotional_state)
        
        spa.ifmax(0.4-spa.dot(myopic_alarm, symbl.PROTOTYPICAL_VIOLENCE),
        symbl.NEUTRAL_OR_POSITIVE >> emotional_state,
        Clean_Up >> judgment)
        
print(model.n_neurons)