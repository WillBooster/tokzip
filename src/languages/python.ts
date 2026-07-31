import { pythonModule } from '../generated/python.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; the model is validated at registration.
registerLanguageModule(pythonModule);

export { pythonModule };
