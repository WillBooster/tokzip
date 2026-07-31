import { dartModule } from '../generated/dart.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; the model is validated at registration.
registerLanguageModule(dartModule);

export { dartModule };
