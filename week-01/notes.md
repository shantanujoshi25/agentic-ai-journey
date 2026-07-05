Bedrock has both per-minute (TPM) and per-day (TPD) quotas. Switch model or geo before assuming code is broken.


What is an inference profile and why does Bedrock force you to use one?
ModelId is inference profile. This is used so the model can route the request according to available region. We do mention the region explicitly but that is converse API region where API request first hits, then internally routes to specifics available region. 

What's the difference between InvokeModel and Converse? Which should you use?
InvokeModel is for traditional model invoking which is specific for models and require rewriting payloads. ConverseAPI is a layer of bedrock that handles all the models and provide a standard and unified model payload.  

What region am I in and why?
us-east-1


